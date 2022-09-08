# import torch
# from torch import nn
# from torch.nn import CrossEntropyLoss, MSELoss
# import torch.nn.functional as F
# from transformers import BartModel, BartConfig, BartPretrainedModel
# from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, shift_tokens_right, _expand_mask
# from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding, BartEncoderLayer, BartClassificationHead
# from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
# import math
# import random
# from typing import Optional, Tuple
#
#
# class BartEncoderPlus(BartPretrainedModel):
#     """
#     Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
#     :class:`BartEncoderLayer`.
#
#     Args:
#         config: BartConfig
#         embed_tokens (torch.nn.Embedding): output embedding
#     """
#
#     def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
#         super().__init__(config)
#
#         self.dropout = config.dropout
#         self.layerdrop = config.encoder_layerdrop
#
#         embed_dim = config.d_model
#         self.padding_idx = config.pad_token_id
#         self.max_source_positions = config.max_position_embeddings
#         self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
#
#         if embed_tokens is not None:
#             self.embed_tokens = embed_tokens
#         else:
#             self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
#
#         self.embed_cates = nn.Embedding(config.cate_size, embed_dim, self.padding_idx)
#
#         self.embed_positions = BartLearnedPositionalEmbedding(
#             config.max_position_embeddings,
#             embed_dim,
#         )
#
#         self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
#         self.layernorm_embedding = nn.LayerNorm(embed_dim)
#
#         self.init_weights()
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         cate_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         Args:
#             input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
#                 Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
#                 provide it.
#
#                 Indices can be obtained using :class:`~transformers.BartTokenizer`. See
#                 :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
#                 for details.
#
#                 `What are input IDs? <../glossary.html#input-ids>`__
#             attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#                 Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
#
#                 - 1 for tokens that are **not masked**,
#                 - 0 for tokens that are **masked**.
#
#                 `What are attention masks? <../glossary.html#attention-mask>`__
#             head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
#                 Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:
#
#                 - 1 indicates the head is **not masked**,
#                 - 0 indicates the heas is **masked**.
#
#             inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
#                 Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
#                 representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
#                 into associated vectors than the model's internal embedding lookup matrix.
#             output_attentions (:obj:`bool`, `optional`):
#                 Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
#                 returned tensors for more detail.
#             output_hidden_states (:obj:`bool`, `optional`):
#                 Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
#                 for more detail.
#             return_dict (:obj:`bool`, `optional`):
#                 Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         # retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#             input_ids = input_ids.view(-1, input_shape[-1])
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")
#
#         if inputs_embeds is None:
#             inputs_embeds = (self.embed_tokens(input_ids) + self.embed_cates(cate_ids)) * self.embed_scale
#
#         embed_pos = self.embed_positions(input_shape)
#
#         hidden_states = inputs_embeds + embed_pos
#         hidden_states = self.layernorm_embedding(hidden_states)
#         hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
#
#         # expand attention_mask
#         if attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)
#
#         encoder_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None
#
#         # check if head_mask has a correct number of layers specified if desired
#         if head_mask is not None:
#             assert head_mask.size()[0] == (
#                 len(self.layers)
#             ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
#         for idx, encoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 encoder_states = encoder_states + (hidden_states,)
#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             dropout_probability = random.uniform(0, 1)
#             if self.training and (dropout_probability < self.layerdrop):  # skip the layer
#                 layer_outputs = (None, None)
#             else:
#                 if getattr(self.config, "gradient_checkpointing", False) and self.training:
#
#                     def create_custom_forward(module):
#                         def custom_forward(*inputs):
#                             return module(*inputs, output_attentions)
#
#                         return custom_forward
#
#                     layer_outputs = torch.utils.checkpoint.checkpoint(
#                         create_custom_forward(encoder_layer),
#                         hidden_states,
#                         attention_mask,
#                         (head_mask[idx] if head_mask is not None else None),
#                     )
#                 else:
#                     layer_outputs = encoder_layer(
#                         hidden_states,
#                         attention_mask,
#                         layer_head_mask=(head_mask[idx] if head_mask is not None else None),
#                         output_attentions=output_attentions,
#                     )
#
#                 hidden_states = layer_outputs[0]
#
#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)
#
#         if output_hidden_states:
#             encoder_states = encoder_states + (hidden_states,)
#
#         if not return_dict:
#             return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
#         )
#
#
# class BartModelPlus(BartPretrainedModel):
#     def __init__(self, config: BartConfig):
#         super().__init__(config)
#
#         padding_idx, vocab_size = config.pad_token_id, config.vocab_size
#         self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
#         self.encoder = BartEncoderPlus(config, self.shared)
#         self.decoder = BartDecoder(config, self.shared)
#
#         self.init_weights()
#
#     def get_input_embeddings(self):
#         return self.shared
#
#     def set_input_embeddings(self, value):
#         self.shared = value
#         self.encoder.embed_tokens = self.shared
#         self.decoder.embed_tokens = self.shared
#
#     def get_encoder(self):
#         return self.encoder
#
#     def get_decoder(self):
#         return self.decoder
#
#     # @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
#     # @add_code_sample_docstrings(
#     #     tokenizer_class=_TOKENIZER_FOR_DOC,
#     #     checkpoint=_CHECKPOINT_FOR_DOC,
#     #     output_type=Seq2SeqModelOutput,
#     #     config_class=_CONFIG_FOR_DOC,
#     # )
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         cate_ids=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         encoder_outputs=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#
#         # different to other models, Bart automatically creates decoder_input_ids from
#         # input_ids if no decoder_input_ids are provided
#         if decoder_input_ids is None and decoder_inputs_embeds is None:
#             decoder_input_ids = shift_tokens_right(
#                 input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
#             )
#
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         if encoder_outputs is None:
#             encoder_outputs = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#         # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )
#
#         # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             encoder_hidden_states=encoder_outputs[0],
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             encoder_head_mask=head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         if not return_dict:
#             return decoder_outputs + encoder_outputs
#
#         return Seq2SeqModelOutput(
#             last_hidden_state=decoder_outputs.last_hidden_state,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )
#
#
# class BartForConditionalGenerationPlus(BartPretrainedModel):
#     base_model_prefix = "model"
#     _keys_to_ignore_on_load_missing = [
#         r"final_logits_bias",
#         r"encoder\.version",
#         r"decoder\.version",
#         r"lm_head\.weight",
#     ]
#
#     def __init__(self, config: BartConfig):
#         super().__init__(config)
#         self.model = BartModelPlus(config)
#         self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
#         self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
#
#         self.init_weights()
#
#     def get_encoder(self):
#         return self.model.get_encoder()
#
#     def get_decoder(self):
#         return self.model.get_decoder()
#
#     def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
#         new_embeddings = super().resize_token_embeddings(new_num_tokens)
#         self._resize_final_logits_bias(new_num_tokens)
#         return new_embeddings
#
#     def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
#         old_num_tokens = self.final_logits_bias.shape[-1]
#         if new_num_tokens <= old_num_tokens:
#             new_bias = self.final_logits_bias[:, :new_num_tokens]
#         else:
#             extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
#             new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
#         self.register_buffer("final_logits_bias", new_bias)
#
#     def get_output_embeddings(self):
#         return self.lm_head
#
#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings
#
#     # @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
#     # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
#     # @add_end_docstrings(BART_GENERATION_EXAMPLE)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         encoder_outputs=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
#             config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
#             (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
#         Returns:
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         if labels is not None:
#             if decoder_input_ids is None:
#                 decoder_input_ids = shift_tokens_right(
#                     labels, self.config.pad_token_id, self.config.decoder_start_token_id
#                 )
#
#         outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             encoder_outputs=encoder_outputs,
#             decoder_attention_mask=decoder_attention_mask,
#             head_mask=head_mask,
#             decoder_head_mask=decoder_head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             decoder_inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
#
#         masked_lm_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
#
#         if not return_dict:
#             output = (lm_logits,) + outputs[1:]
#             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
#
#         return Seq2SeqLMOutput(
#             loss=masked_lm_loss,
#             logits=lm_logits,
#             past_key_values=outputs.past_key_values,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )
#
#     def prepare_inputs_for_generation(
#         self,
#         decoder_input_ids,
#         past=None,
#         attention_mask=None,
#         head_mask=None,
#         use_cache=None,
#         encoder_outputs=None,
#         **kwargs
#     ):
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             decoder_input_ids = decoder_input_ids[:, -1:]
#
#         return {
#             "input_ids": None,  # encoder_outputs is defined. input_ids not needed
#             "encoder_outputs": encoder_outputs,
#             "past_key_values": past,
#             "decoder_input_ids": decoder_input_ids,
#             "attention_mask": attention_mask,
#             "head_mask": head_mask,
#             "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
#         }
#
#     def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
#         return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
#
#     @staticmethod
#     def _reorder_cache(past, beam_idx):
#         reordered_past = ()
#         for layer_past in past:
#             # cached cross_attention states don't have to be reordered -> they are always the same
#             reordered_past += (
#                 tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
#             )
#         return reordered_past
