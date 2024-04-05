import torch
import torch.nn as nn
import numpy as np
from transformers import MT5ForConditionalGeneration, MBartForConditionalGeneration, AutoConfig
from .model_help import MBartCopy, MT5Copy
from .utils import lang_map
import ipdb

# for constrained decoding
class Prefix_fn_cls():
    def __init__(self, tokenizer, special_tokens, input_enc_idxs):
        self.tokenizer=tokenizer
        self.input_enc_idxs=input_enc_idxs
        self.special_ids = [element for l in self.tokenizer(special_tokens, add_special_tokens=False)['input_ids'] for element in l]
    def get(self, batch_id, previous_token):
        # get input
        inputs = list(set(self.input_enc_idxs[batch_id].tolist()+self.special_ids))
        return inputs

class XGearEAEModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        
        if self.config.pretrained_model_name.startswith('google/mt5-'):
            self.model = MT5ForConditionalGeneration.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
        elif self.config.pretrained_model_name.startswith("facebook/mbart-large-50"):
            self.model = MBartForConditionalGeneration.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
        else:
            raise ValueError("Not implemented.")
            
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def process_data(self, batch):
        if self.config.pretrained_model_name.startswith("facebook/mbart-large-50"):
            enc_idxs = []
            enc_attn = []
            max_len = 0
            for passage_, lang in zip(batch.batch_input, batch.batch_language):
                # set tokenizer language
                self.tokenizer.src_lang = lang_map[lang]
                passage_input = self.tokenizer(passage_, return_tensors='pt')
                enc_idx = passage_input['input_ids'][0]
                enc_att = [1]*enc_idx.size(0)
                enc_idxs.append(enc_idx)
                enc_attn.append(enc_att)
                if max_len < enc_idx.size(0):
                    max_len = enc_idx.size(0)

            enc_idxs = [torch.cat((x, torch.LongTensor([self.tokenizer.pad_token_id]*(max_len-x.size(0)))), dim=0) for x in enc_idxs]
            enc_attn = [x + [0]*(max_len-len(x)) for x in enc_attn]         
            enc_idxs = torch.stack(enc_idxs, dim=0)
            enc_attn = torch.LongTensor(enc_attn)
        else:
            # encoder inputs
            inputs = self.tokenizer(batch.batch_input, return_tensors='pt', padding=True)
            enc_idxs = inputs['input_ids']
            enc_attn = inputs['attention_mask']

        # decoder inputs
        targets = self.tokenizer(batch.batch_target, return_tensors='pt', padding=True)
        batch_size = enc_idxs.size(0)
        
        if "t5" in self.config.pretrained_model_name:
            padding = torch.ones((batch_size, 1), dtype=torch.long)
            padding[:] = self.tokenizer.pad_token_id
            dec_idxs = torch.cat((padding, targets['input_ids']), dim=1)
            dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), targets['attention_mask']), dim=1)
        else:
            dec_idxs = targets['input_ids']
            dec_idxs[:,0] = self.tokenizer.eos_token_id
            dec_attn = targets['attention_mask']
            # padding = torch.ones((batch_size, 1), dtype=torch.long)
            # padding[:] = self.tokenizer.lang_code_to_id[lang_map[lang]]
            # dec_idxs = torch.cat((padding, targets['input_ids']), dim=1)
            # dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), targets['attention_mask']), dim=1)
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        raw_lbl_idxs = raw_lbl_idxs.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        return enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs

    def forward(self, batch):
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs = self.process_data(batch)
        outputs = self.model(input_ids=enc_idxs, 
                             attention_mask=enc_attn, 
                             decoder_input_ids=dec_idxs, 
                             decoder_attention_mask=dec_attn, 
                             labels=lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        # all_loss = torch.nn.CrossEntropyLoss(reduction='none')
        # loss_raw = all_loss(outputs['logits'].view(-1, outputs['logits'].shape[-1]), lbl_idxs.view(-1)).view(outputs['logits'].shape[0], -1)
        # assert round((loss_raw.sum() / (lbl_idxs != -100).sum()).item(), 2) == round(loss.item(), 2), (ound((loss_raw.sum() / (lbl_idxs != -100).sum()).item(), 2), round(loss.item(), 2))
        # loss_raw = torch.div(loss_raw.sum(dim=-1), (lbl_idxs != -100).sum(dim=-1))
        
        return loss
        # return loss, loss_raw
        
    def predict(self, batch, num_beams=4, max_length=50, constrained_decode=False, special_tokens=[]):
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs = self.process_data(batch)
        return self.generate(enc_idxs, enc_attn, num_beams, max_length, constrained_decode=constrained_decode, special_tokens=special_tokens)
    
    def generate(self, input_ids, attention_mask, num_beams=4, max_length=50, constrained_decode=False, special_tokens=[], **kwargs):
        self.eval()
        with torch.no_grad():
            if num_beams == 1:
                self.model._cache_input_ids = input_ids
            else:
                expanded_return_idx = (
                    torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(input_ids.device)
                )
                expanded_input_ids = input_ids.index_select(0, expanded_return_idx)
                self.model._cache_input_ids = expanded_input_ids
            
            if constrained_decode:
                prefix_fn_obj = Prefix_fn_cls(self.tokenizer, special_tokens, input_ids)
                outputs = self.model.generate(input_ids=input_ids,
                        attention_mask=attention_mask, 
                        num_beams=num_beams, 
                        max_length=max_length,
                        forced_bos_token_id=None,
                        prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                    ) 
            else:
                outputs = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          num_beams=num_beams, 
                                          max_length=max_length, 
                                          forced_bos_token_id=None)
        final_output, raw_output = [], []
        for bid in range(len(input_ids)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            raw_sentence = self.tokenizer.convert_ids_to_tokens(outputs[bid])
            final_output.append(output_sentence)
            raw_output.append(raw_sentence)
        # import ipdb; ipdb.set_trace()
        self.train()
        return final_output, raw_output

class XGearCopyEAEModel(XGearEAEModel):
    def __init__(self, config, tokenizer, type_set):
        super().__init__(config, tokenizer, type_set)
        
        if self.config.pretrained_model_name.startswith('google/mt5-'):
            self.model = MT5Copy.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, output_attentions=True)
        elif self.config.pretrained_model_name.startswith("facebook/mbart-large-50"):
            self.model = MBartCopy.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, output_attentions=True)
        else:
            raise ValueError("Not implemented.")  
        self.model.resize_token_embeddings(len(self.tokenizer))