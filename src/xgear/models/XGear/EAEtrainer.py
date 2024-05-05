import os, sys, logging, tqdm, pprint, json
from pathlib import Path
import torch
import numpy as np
import random
from collections import namedtuple
from transformers import MBart50Tokenizer, MT5Tokenizer, AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ..trainer import BasicTrainer
from .EAEmodel import XGearEAEModel, XGearCopyEAEModel
from .template_generate import event_template, eve_template_generator, IN_SEP, NO_ROLE, AND, TAGGER
from .pattern import patterns
from .utils import lang_map, get_span_idx, get_span_idxs_zh
from xgear.scorer import compute_EAE_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

EAEBatch_fields = ['batch_doc_id', 'batch_wnd_id', 'batch_tokens', 'batch_text', 'batch_piece_idxs', 
                   'batch_token_start_idxs', 'batch_trigger', 'batch_arguments', 'batch_input', 
                   'batch_target', 'batch_added_sequence', 'batch_passage', 'batch_language']
EAEBatch = namedtuple('EAEBatch', field_names=EAEBatch_fields, defaults=[None] * len(EAEBatch_fields))
EAETransBatch = namedtuple('EAETransBatch', field_names=['batch_input', 'batch_target', 'batch_language'], defaults=[None] * 3)

def EAE_collate_fn(batch):
    return EAEBatch(
        batch_language=[instance["language"] for instance in batch],
        batch_doc_id=[instance.get("doc_id", None) for instance in batch],
        batch_wnd_id=[instance["wnd_id"] for instance in batch],
        batch_tokens=[instance.get("tokens", None) for instance in batch], 
        batch_text=[instance.get("text", None) for instance in batch], 
        batch_piece_idxs=[instance.get("piece_idxs", None) for instance in batch], 
        batch_token_start_idxs=[instance.get("token_start_idxs", None) for instance in batch], 
        batch_trigger=[instance.get("trigger", None) for instance in batch], 
        batch_arguments=[instance.get("arguments", None) for instance in batch], 
        batch_input=[instance["input"] for instance in batch], 
        batch_target=[instance["target"] for instance in batch], 
        batch_added_sequence=[instance.get("added_sequence", None) for instance in batch], 
        batch_passage=[instance.get("passage", None) for instance in batch], 
    )

def EAE_trans_collate_fn(batch):
    return EAETransBatch(
        batch_input=[instance["input"] for instance in batch], 
        batch_target=[instance["target"] for instance in batch], 
        batch_language=[instance["language"] for instance in batch], 
    )

def get_special_tokens(config):
    special_tokens = []
    sep_tokens = []
    if "triggerword" in config.input_style:
        sep_tokens += [IN_SEP["triggerword"]]
    if "template" in config.input_style:
        sep_tokens += [IN_SEP["template"]]
    if "trigger_tagger" in config.input_style:
        special_tokens += list(TAGGER.values())
    if "argument:roletype" in config.output_style:
        role_list = sorted(set([role for eve_type_info in patterns[config.dataset].values() for role in eve_type_info['valid roles']]))
        logger.info(f"All the roles are: {role_list}")
        special_tokens += [f"<--{r}-->" for r in role_list]
        special_tokens += [f"</--{r}-->" for r in role_list]
        special_tokens += [NO_ROLE, AND]
    return special_tokens, sep_tokens

class XGearEAETrainer(BasicTrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        self.tokenizer = None
        self.model = None

    @classmethod
    def add_extra_info_fn(cls, instances, raw_data, config):
        with open(config.extra_info_file, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        aux_data = [json.loads(line) for line in lines]

        with open(config.postag_type_stoi_file, 'r', encoding='utf-8') as fp:
            postag_type_stoi = json.load(fp)

        extra_info_map = {}
        for aux_data_ in aux_data:
            doc_id = aux_data_['doc_id']
            wnd_id = aux_data_['wnd_id']
            postags = aux_data_['postags']
            postags = [postag_type_stoi[postag] for postag in postags]
            dep_dist_matrix = aux_data_['dep dist matrix']
            stanza_result = aux_data_['stanza result']
            extra_info_map[(doc_id, wnd_id)] = {
                "postags": postags,
                "dep dist matrix": dep_dist_matrix,
                "stanza result": stanza_result
            }

        for instance in instances:
            instance["extra_info"] = extra_info_map[(instance["doc_id"], instance["wnd_id"])]
        
        return instances
    
    def load_model(self, checkpoint=None):
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            state = torch.load(checkpoint, map_location=f'cuda:{self.config.gpu_device}')
            self.tokenizer = state["tokenizer"]
            self.type_set = state["type_set"]
            self.model = XGearEAEModel(self.config, self.tokenizer, self.type_set)
            self.model.load_state_dict(state['model'])
            self.model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('google/mt5-'):
                self.tokenizer = MT5Tokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            if self.config.pretrained_model_name.startswith('facebook/mbart-large-50'):
                self.tokenizer = MBart50Tokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
                # We do not assign language first, but when using, we need to notify the model.
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, use_fast=False)
            
            # check valid styles
            assert np.all([style in ['triggerword', 'template', 'trigger_tagger'] for style in self.config.input_style])
            assert np.all([style in ['argument:roletype'] for style in self.config.output_style])

            special_tokens, sep_tokens = get_special_tokens(self.config)
            logger.info(f"Add tokens {special_tokens+sep_tokens}")
            self.tokenizer.add_tokens(special_tokens+sep_tokens)
            
            self.model = XGearEAEModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)
            
        self.generate_vocab()
            
    def generate_vocab(self):
        
        event_type_itos = sorted(self.type_set["trigger"])
        event_type_stoi = {e: i for i, e in enumerate(event_type_itos)}
        role_type_itos = sorted(self.type_set["role"])
        role_type_stoi = {r: i for i, r in enumerate(role_type_itos)}
        self.vocab = {"event_type_itos": event_type_itos, 
                    "event_type_stoi": event_type_stoi,
                    "role_type_itos": role_type_itos,
                    "role_type_stoi": role_type_stoi,
                    }
    
    def process_data(self, data):
        assert self.tokenizer, "Please load model and tokneizer before processing data!"

        n_total = 0
        new_data = []
        for dt in data:
            n_total += 1
            _trigger = (dt["trigger"][0], dt["trigger"][1], dt["trigger"][2])
            _arguments = [(_trigger, (r[0], r[1], r[2])) for r in dt["arguments"]]
            event_template = eve_template_generator(self.config.dataset, dt["tokens"], [_trigger], _arguments, self.config.input_style, self.config.output_style, dt['language'])
            event_training_data = event_template.get_training_data()
            assert len(event_training_data) == 1
            
            data_ = event_training_data[0]

            self.tokenizer.src_lang = lang_map[dt['language']]
            if len(self.tokenizer.tokenize(data_["seq_in"])) > self.config.max_length:
                continue
                
            pieces = [self.tokenizer.tokenize(t) for t in dt["tokens"]]
            token_lens = [len(p) for p in pieces]
            pieces = [p for piece in pieces for p in piece]
            piece_idxs = self.tokenizer.convert_tokens_to_ids(pieces)
            assert sum(token_lens) == len(piece_idxs)
            token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]

            new_dt = {"doc_id": dt["doc_id"], 
                      "wnd_id": dt["wnd_id"], 
                      "tokens": dt["tokens"], 
                      "text": dt["text"], 
                      "piece_idxs": piece_idxs, 
                      "token_start_idxs": token_start_idxs,
                      "trigger": dt["trigger"], 
                      "arguments": dt["arguments"], 
                      "input": data_["seq_in"], 
                      "target": data_["seq_out"], 
                      "info": data_["gold_event"], # with field "trigger text", "trigger span", "event type", "arguments", "passage", "tokens",
                      "added_sequence": data_["added_sequence"],
                      "passage": data_["passage"],
                      "language": dt["language"],
                     }
            new_data.append(new_dt)
                
        logger.info(f"Generate {len(new_data)} XGear EAE instances from {n_total} EAE instances")

        return new_data

    def train(self, train_data, dev_data, trans_data=None):
        if hasattr(self.config, "finetune") and hasattr(self.config, "trained_model_dir") and self.config.finetune: 
            self.load_model(checkpoint=self.config.trained_model_dir)   # might need to provide exact path
        else:
            self.load_model()

        if type(train_data[0]) == list:
            internal_train_data = []
            for train_data_ in train_data:
                internal_train_data.extend(self.process_data(train_data_)) # assume that training data can be trained together
        else:
            internal_train_data = self.process_data(train_data)

        if type(dev_data[0]) == list:
            internal_dev_datas = []
            for dev_data_ in dev_data:
                internal_dev_datas.append(self.process_data(dev_data_)) # assume that the dev data is used to save different model
        else:
            internal_dev_datas = [self.process_data(dev_data)]

        """
        If you want to include generation-based pairs, you can create the data here and integrate them into the internal_train_data.
        """
        final_trans_data = []
        if hasattr(self.config, "no_source") and self.config.no_source:
            internal_train_data = []
        if trans_data is not None:
            internal_trans_data = {}
            for lang in trans_data:
                internal_trans_data[lang] = []
                assert len(trans_data[lang]["train_input"]) == len(trans_data[lang]["train_output"])
                for inp, out in zip(trans_data[lang]["train_input"], trans_data[lang]["train_output"]):
                    internal_trans_data[lang].append({
                        "input": inp,
                        "target": out,
                        "language": lang
                    })
                if hasattr(self.config, "num_sample") and self.config.num_sample > 0:
                    final_trans_data.extend(random.sample(internal_trans_data[lang], self.config.num_sample))
                else:
                    final_trans_data.extend(internal_trans_data[lang])

        param_groups = [{'params': self.model.parameters(), 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay}]
        optimizer = AdamW(params=param_groups)
        
        final_internal_data = internal_train_data + final_trans_data
        train_batch_num = len(final_internal_data) // self.config.train_batch_size + (len(final_internal_data) % self.config.train_batch_size != 0)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=train_batch_num*self.config.warmup_epoch,
                                            num_training_steps=train_batch_num*self.config.max_epoch)
        
        best_scores = []
        best_epochs = []
        for _ in range(len(internal_dev_datas)):
            best_scores.append({"argument_cls": {"f1": 0.0}})
            best_epochs.append(-1)
        
        with open(os.path.join(self.config.output_dir, "final_internal_data_train.json"), 'w') as fw:
            json.dump(final_internal_data, fw, indent=2, ensure_ascii=False)
        with open(os.path.join(self.config.output_dir, "final_trans_data_train.json"), 'w') as fw:
            json.dump(final_trans_data, fw, indent=2, ensure_ascii=False)
        
        for epoch in range(1, self.config.max_epoch+1):
            logger.info(f"Log path: {self.config.log_path}")
            logger.info(f"Epoch {epoch}")

            # training step
            progress = tqdm.tqdm(total=train_batch_num, ncols=100, desc='Train {}'.format(epoch))

            self.model.train()
            optimizer.zero_grad()
            cummulate_loss = []
            # cummulate_loss_langs = {}

            for batch_idx, batch in enumerate(DataLoader(final_internal_data, batch_size=self.config.train_batch_size // self.config.accumulate_step, 
                                                         shuffle=True, drop_last=False, collate_fn=EAE_trans_collate_fn)):

                loss = self.model(batch)
                # loss, loss_raw = self.model(batch)
                loss = loss * (1 / self.config.accumulate_step)
                cummulate_loss.append(loss.item())
                loss.backward()

                # # Language specific loss
                # all_langs_in_batch = list(set(batch.batch_language))
                # loss_raw = loss_raw * (1 / self.config.accumulate_step)
                # for l in all_langs_in_batch:
                #     cond = [ True if b == l else False for b in batch.batch_language ]
                #     loss_l = (loss_raw.cpu() * torch.tensor(cond)).sum() / len(cond)
                #     # loss_l = (loss_raw * torch.tensor(cond).cuda()).sum() / len(batch.batch_language)
                #     if l not in cummulate_loss_langs:
                #         cummulate_loss_langs[l] = []
                #     cummulate_loss_langs[l].append(loss_l.item())

                if (batch_idx + 1) % self.config.accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clipping)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                        
            progress.close()
            logger.info(f"Average training loss: {np.mean(cummulate_loss)}")
            # for l in cummulate_loss_langs:
            #     logger.info(f"Average training {l} loss: {np.mean(cummulate_loss_langs[l])}")

            # Shuffle trans sampled data
            if hasattr(self.config, "sample_shuffle") and self.config.sample_shuffle and hasattr(self.config, "num_sample") and self.config.num_sample > 0:
                final_trans_data = []
                for lang in internal_trans_data:
                    final_trans_data.extend(random.sample(internal_trans_data[lang], self.config.num_sample))
                final_internal_data = internal_train_data + final_trans_data
            
            if epoch > 8:
                # eval dev
                for dev_idx, internal_dev_data in enumerate(internal_dev_datas):
                    predictions = self.internal_predict(internal_dev_data, split="Dev set {}".format(dev_idx))
                    dev_scores = compute_EAE_scores(predictions, internal_dev_data, metrics={"argument_id", "argument_cls", "argument_attached_id", "argument_attached_cls"})

                    # print scores
                    print(f"Dev {dev_idx} Epoch: {epoch}")
                    print_scores(dev_scores)
                    
                    # # save steps for analysis
                    # if epoch % 5 == 0:
                    #     logger.info("Saving model for dev set {}...".format(dev_idx))
                    #     state = dict(model=self.model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                    #     torch.save(state, os.path.join(self.config.output_dir, "model_dev{}_epoch{}.state".format(dev_idx, epoch)))
                    
                    if dev_scores["argument_cls"]["f1"] >= best_scores[dev_idx]["argument_cls"]["f1"]:
                        logger.info("Saving best model for dev set {}...".format(dev_idx))
                        state = dict(model=self.model.state_dict(), tokenizer=self.tokenizer, type_set=self.type_set)
                        torch.save(state, os.path.join(self.config.output_dir, "best_model_dev{}.state".format(dev_idx)))
                        best_scores[dev_idx] = dev_scores
                        best_epochs[dev_idx] = epoch
                        with open(os.path.join(self.config.output_dir, "prediction_on_dev{}.json".format(dev_idx)), 'w') as fw:
                            json.dump(predictions, fw, indent=2)
                        
                    logger.info(pprint.pformat({"epoch": epoch, "dev_scores": dev_scores}))
                    logger.info(pprint.pformat({"best_epoch": best_epochs[dev_idx], "best_scores": best_scores[dev_idx]}))

        return os.path.join(self.config.output_dir, "best_model_dev{}.state".format(0))
        
    def internal_predict(self, eval_data, split, constrained_decode=False):
        self.model.eval()
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        special_tokens, sep_tokens = get_special_tokens(self.config)
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.eval_batch_size, 
                                                     shuffle=False, drop_last=False, collate_fn=EAE_collate_fn)):
            progress.update(1)
            pred_texts, raw_outputs = self.model.predict(batch, num_beams=self.config.beam_size, max_length=self.config.max_output_length,
                            constrained_decode=constrained_decode, special_tokens=special_tokens+sep_tokens)
            for idx, (doc_id, wnd_id, tokens, text, piece_idxs, token_start_idxs, trigger, pred_text, raw_text, eval_lang) in enumerate(zip(batch.batch_doc_id, 
                    batch.batch_wnd_id, batch.batch_tokens, batch.batch_text, batch.batch_piece_idxs, batch.batch_token_start_idxs, 
                    batch.batch_trigger, pred_texts, raw_outputs, batch.batch_language)):
                
                template = event_template(trigger[2], patterns[self.config.dataset][trigger[2]], self.config.input_style, self.config.output_style, tokens, eval_lang)
                pred_objects = template.decode(pred_text)
                
                pred_arguments = []
                for span, role_type, _ in pred_objects:
                    if eval_lang == "chinese":
                        sid, eid = get_span_idxs_zh(tokens, span, trigger_span=trigger[:2])
                    else:
                        sid, eid = get_span_idx(piece_idxs, token_start_idxs, span, self.tokenizer, trigger_span=trigger[:2])
                    if sid == -1:
                        continue
                    pred_arguments.append((sid, eid, role_type, span))
                
                prediction = {"doc_id": doc_id,  
                              "wnd_id": wnd_id, 
                              "tokens": tokens, 
                              "text": text, 
                              "trigger": trigger, 
                              "arguments": pred_arguments,
                              "pred_text": pred_text,
                              "raw_text": raw_text
                             }
                predictions.append(prediction)

        progress.close()
        
        return predictions
    
    def internal_predict_seq(self, eval_data, split="Dev", constrained_decode=False):
        self.model.eval()
        eval_batch_num = len(eval_data) // self.config.eval_batch_size + (len(eval_data) % self.config.eval_batch_size != 0)
        progress = tqdm.tqdm(total=eval_batch_num, ncols=100, desc=split)
        
        predictions = []
        special_tokens, sep_tokens = get_special_tokens(self.config)
        for batch_idx, batch in enumerate(DataLoader(eval_data, batch_size=self.config.eval_batch_size, 
                                                     shuffle=False, drop_last=False, collate_fn=EAE_trans_collate_fn)):
            progress.update(1)
            pred_texts, _ = self.model.predict(batch, num_beams=self.config.beam_size, max_length=self.config.max_output_length,
                            constrained_decode=constrained_decode, special_tokens=special_tokens+sep_tokens)
            predictions.extend(pred_texts)
        progress.close()
        return predictions
    
    def predict(self, data, constrained_decode=False, trans_data=False, split="test", lang=None):
        assert self.tokenizer and self.model

        if not trans_data:
            internal_data = self.process_data(data)
            predictions = self.internal_predict(internal_data, split="Test", constrained_decode=constrained_decode)
            return predictions
        else:
            split_input = split + "_input"
            split_output = split + "_output"
            assert split_input in data and split_output in data

            internal_data = []
            assert len(data[split_input]) == len(data[split_output])
            for inp, out in zip(data[split_input], data[split_output]):
                internal_data.append({
                    "input": inp,
                    "target": out,
                    "language": lang
                })
            predictions = self.internal_predict_seq(internal_data, split=split, constrained_decode=constrained_decode)
            return predictions

class XGearCopyEAETrainer(XGearEAETrainer):
    def __init__(self, config, type_set=None):
        super().__init__(config, type_set)
        self.tokenizer = None
        self.model = None
    
    def load_model(self, checkpoint=None):
        if checkpoint:
            logger.info(f"Loading model from {checkpoint}")
            state = torch.load(checkpoint, map_location=f'cuda:{self.config.gpu_device}')
            self.tokenizer = state["tokenizer"]
            self.type_set = state["type_set"]
            self.model = XGearCopyEAEModel(self.config, self.tokenizer, self.type_set)
            self.model.load_state_dict(state['model'])
            self.model.cuda(device=self.config.gpu_device)
        else:
            logger.info(f"Loading model from {self.config.pretrained_model_name}")
            if self.config.pretrained_model_name.startswith('google/mt5-'):
                self.tokenizer = MT5Tokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
            if self.config.pretrained_model_name.startswith('facebook/mbart-large-50'):
                self.tokenizer = MBart50Tokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir)
                # We do not assign language first, but when using, we need to notify the model.
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name, cache_dir=self.config.cache_dir, use_fast=False)
            
            # check valid styles
            assert np.all([style in ['triggerword', 'template', 'trigger_tagger'] for style in self.config.input_style])
            assert np.all([style in ['argument:roletype'] for style in self.config.output_style])

            special_tokens = []
            sep_tokens = []
            if "triggerword" in self.config.input_style:
                sep_tokens += [IN_SEP["triggerword"]]
            if "template" in self.config.input_style:
                sep_tokens += [IN_SEP["template"]]
            if "trigger_tagger" in self.config.input_style:
                special_tokens += list(TAGGER.values())
            if "argument:roletype" in self.config.output_style:
                role_list = sorted(set([role for eve_type_info in patterns[self.config.dataset].values() for role in eve_type_info['valid roles']]))
                logger.info(f"All the roles are: {role_list}")
                special_tokens += [f"<--{r}-->" for r in role_list]
                special_tokens += [f"</--{r}-->" for r in role_list]
                special_tokens += [NO_ROLE, AND]
            logger.info(f"Add tokens {special_tokens+sep_tokens}")
            self.tokenizer.add_tokens(special_tokens+sep_tokens)
            
            self.model = XGearCopyEAEModel(self.config, self.tokenizer, self.type_set)
            self.model.cuda(device=self.config.gpu_device)
            
        self.generate_vocab()