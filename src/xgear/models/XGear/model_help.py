from transformers import MT5ForConditionalGeneration, MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from torch.nn import CrossEntropyLoss, NLLLoss
import torch, logging
import torch.nn as nn
import ipdb

logger = logging.getLogger(__name__)

class MBartCopy(MBartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = ["linear_copy.weight", "linear_copy.bias"]
    def __init__(self, config):
        super().__init__(config)
        self.linear_copy = nn.Linear(config.d_model, 1)
        self.counter = 0
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        
        if input_ids is None:
            input_ids = self._cache_input_ids # batch x sequence_length
        try:
            assert input_ids.size(0) == outputs['encoder_last_hidden_state'].size(0)
        except:
            ipdb.set_trace()

        # # Copy distribution
        cross_attentions = outputs['cross_attentions']
        # This is in tuple format, and each of them is of shape (batch_size, num_heads, dec_sequence_length, enc_sequence_length).
        # This are the attentions weights of the decoder’s cross-attention layer, after the attention softmax.
        cross_attentions = torch.stack(cross_attentions[-1:], dim=1) # TODO: we can change the used layer here.
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate layers
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate heads
        # Now, "cross attentions" is of shape (batch_size, dec_sequence_length, enc_sequence_length)

        # Probability of copying
        p_gen = torch.sigmoid(self.linear_copy(outputs['last_hidden_state']))
        
        # Merge distribution
        original_word_pro = torch.softmax(lm_logits, dim=-1) * p_gen #[batch, sequence_length, vocab_size]
        copy_words = input_ids.unsqueeze(1).repeat(1, cross_attentions.size(1), 1) #(batch, target_length, encoder_length)
        lm_logits = torch.scatter_add(original_word_pro, 2, copy_words, cross_attentions*(1-p_gen))

        eps = 1e-25
        lm_logits = torch.log(lm_logits+eps)
        # lm_logits = torch.log(lm_logits)

        masked_lm_loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss_fct = NLLLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class MT5Copy(MT5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = ["linear_copy.weight", "linear_copy.bias"]
    def __init__(self, config):
        super().__init__(config)
        self.linear_copy = nn.Linear(self.model_dim, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        if input_ids is None:
            input_ids = self._cache_input_ids # batch x sequence_length
        try:
            assert input_ids.size(0) == hidden_states.size(0)
        except:
            ipdb.set_trace()  

        # Copy distribution
        cross_attentions = decoder_outputs['cross_attentions']
        # This is in tuple format, and each of them is of shape (batch_size, num_heads, dec_sequence_length, enc_sequence_length).
        # This are the attentions weights of the decoder’s cross-attention layer, after the attention softmax.
        cross_attentions = torch.stack(cross_attentions[-1:], dim=1) # TODO: we can change the used layer here.
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate layers
        cross_attentions = torch.mean(cross_attentions, dim=1) # aggregate heads
        # Now, "cross attentions" is of shape (batch_size, dec_sequence_length, enc_sequence_length)

        # Probability of copying
        p_gen = torch.sigmoid(self.linear_copy(sequence_output))
        
        # Merge distribution
        original_word_pro = torch.softmax(lm_logits, dim=-1) * p_gen #[batch, sequence_length, vocab_size]
        copy_words = input_ids.unsqueeze(1).repeat(1, cross_attentions.size(1), 1) #(batch, target_length, encoder_length)
        lm_logits = torch.scatter_add(original_word_pro, 2, copy_words, cross_attentions*(1-p_gen))
        
        eps = 1e-20
        lm_logits = torch.log(lm_logits+eps)
        # lm_logits = torch.log(lm_logits)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss_fct = NLLLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )