import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BartModel, BartConfig, BartPretrainedModel
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import TokenClassifierOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


class BartForSeqClassificationAndGeneration(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        # seq2seq addidional init
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.insertion_token = None
        # seq_cls additional init
        self.num_labels_tok_cls = config.num_labels_tok_cls
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels_tok_cls)
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def set_insertion_token(self, insertion_token):
        self.insertion_token = insertion_token

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_insert_token_embeddings(self, input_ids, sequence_outputs):
        batch_size, _, model_dim = sequence_outputs.shape
        max_insert_token_num = torch.max(torch.sum((input_ids == self.insertion_token).int(), dim=1), dim=0).values.item()
        insert_token_positions_list = (input_ids == self.insertion_token).nonzero().cpu().numpy()
        insert_token_positions = defaultdict(list)
        # insert_token_mask = torch.ones((batch_size, max_insert_token_num))
        for idx_x, idx_y in insert_token_positions_list:
            insert_token_positions[idx_x].append(idx_y)
        insert_token_embeddings = torch.zeros((batch_size, max_insert_token_num, model_dim))
        for idx_x in range(batch_size):
            # insert_token_mask[idx_x][len(insert_token_positions[idx_x]):] = torch.tensor(float('-inf'))
            for i, idx_y in enumerate(insert_token_positions[idx_x]):
                insert_token_embeddings[idx_x][i] = sequence_outputs[idx_x][idx_y]
        # insert_token_mask = insert_token_mask.unsqueeze(dim=2).repeat(1, 1, model_dim)
        return insert_token_embeddings.to(input_ids.device)

    def get_insertion_prediction(self, logits):
        probs_insertion = nn.functional.softmax(logits, dim=2)[:, :, 1]
        preds_insertion = torch.argmax(probs_insertion, dim=1)
        return preds_insertion, probs_insertion

    def forward_multi(
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
            labels_seq2seq=None,
            labels_tok_cls=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            get_results_seq2seq=True,
            get_results_tok_cls=True,
    ):
        # Copying directly from BartModel
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # End copying from BartModel
        if labels_seq2seq is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels_seq2seq, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        else:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        results_seq2seq, results_tok_cls = None, None
        # Part1: Get seq2seq results
        if get_results_seq2seq:
            outputs_seq2seq = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
            lm_logits = self.lm_head(outputs_seq2seq[0]) + self.final_logits_bias
            masked_lm_loss = None
            if labels_seq2seq is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels_seq2seq.view(-1))
            if not return_dict:
                output = (lm_logits,) + outputs_seq2seq[1:]
                results_seq2seq = ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            else:
                results_seq2seq = Seq2SeqLMOutput(
                    loss=masked_lm_loss,
                    logits=lm_logits,
                    past_key_values=outputs_seq2seq.past_key_values,
                    decoder_hidden_states=outputs_seq2seq.decoder_hidden_states,
                    decoder_attentions=outputs_seq2seq.decoder_attentions,
                    cross_attentions=outputs_seq2seq.cross_attentions,
                    encoder_last_hidden_state=outputs_seq2seq.encoder_last_hidden_state,
                    encoder_hidden_states=outputs_seq2seq.encoder_hidden_states,
                    encoder_attentions=outputs_seq2seq.encoder_attentions,
                )
        # Part2: Get seq_cls results
        if get_results_tok_cls:
            sequence_outputs = encoder_outputs[0]
            sequence_outputs = self.dropout(sequence_outputs)
            ins_tok_embeddings = self.get_insert_token_embeddings(input_ids, sequence_outputs)
            logits = self.classifier(ins_tok_embeddings)
            loss = None
            if labels_tok_cls is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                # if attention_mask is not None:
                active_loss = labels_tok_cls.view(-1) != -100
                active_logits = logits.view(-1, self.num_labels_tok_cls)
                active_labels = torch.where(
                    active_loss, labels_tok_cls.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels_tok_cls)
                )
                loss = loss_fct(active_logits, active_labels)
                # else:
                #     loss = loss_fct(logits.view(-1, self.num_labels), labels_tok_cls.view(-1))
            if not return_dict:
                results = (logits,) + encoder_outputs[2:]
                results_tok_cls = ((loss,) + results) if loss is not None else sequence_outputs
            else:
                results_tok_cls = TokenClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )
        return results_seq2seq, results_tok_cls

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
            labels_seq2seq=None,
            labels_tok_cls=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        results_seq2seq, _ = self.forward_multi(input_ids, attention_mask,
                                                              decoder_input_ids, decoder_attention_mask,
                                                              head_mask, decoder_head_mask,
                                                              cross_attn_head_mask, encoder_outputs,
                                                              past_key_values, inputs_embeds,
                                                              decoder_inputs_embeds, labels_seq2seq,
                                                              labels_tok_cls, use_cache,
                                                              output_attentions, output_hidden_states,
                                                              return_dict, get_results_seq2seq=True, get_results_tok_cls=False)
        return results_seq2seq

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


@dataclass
class DataCollatorForClassificationAndGeneration:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels_seq2seq = [feature["labels_seq2seq"] for feature in features]
        labels_tok_cls = [feature["labels_tok_cls"] for feature in features]
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        # if labels is not None:
        # 1. Pad labels_seq2seq
        max_label_seq2seq_length = max(len(l) for l in labels_seq2seq)
        padding_side = self.tokenizer.padding_side
        for feature in features:
            remainder = [self.label_pad_token_id] * (max_label_seq2seq_length - len(feature["labels_seq2seq"]))
            feature["labels_seq2seq"] = (
                feature["labels_seq2seq"] + remainder if padding_side == "right" else remainder + feature["labels_seq2seq"]
            )
        # 2. Pad labels_tok_cls
        max_label_tok_cls_length = max(len(l) for l in labels_tok_cls)
        padding_side = self.tokenizer.padding_side
        for feature in features:
            remainder = [self.label_pad_token_id] * (max_label_tok_cls_length - len(feature["labels_tok_cls"]))
            feature["labels_tok_cls"] = (
                feature["labels_tok_cls"] + remainder if padding_side == "right" else remainder + feature["labels_tok_cls"]
            )
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels_seq2seq"])
            features["decoder_input_ids"] = decoder_input_ids

        return features



