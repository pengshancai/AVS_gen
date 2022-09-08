import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, TokenClassifierOutput
from transformers import BertPreTrainedModel, BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding
from enum import Enum
import torch.nn.functional as F
import random


class BertModelPlus(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.embeddings_plus = nn.Embedding(config.cate_vocab_size, config.hidden_size)
        nn.init.xavier_normal_(self.embeddings_plus.weight)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPoolingAndCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC,
    # )

    def forward(
        self,
        input_ids=None,
        cate_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        word_embeddings = self.embeddings.word_embeddings(input_ids)
        cate_embeddings = self.embeddings_plus(cate_ids)
        inputs_embeds = word_embeddings + cate_embeddings
        embedding_output = self.embeddings(
            # input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForTokenClassificationPlus(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModelPlus(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=TokenClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        cate_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert.forward(
            input_ids,
            cate_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


EncodedInput = List[int]


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            "%r is not a valid %s, please select one of %s"
            % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """
    Possible values for the ``return_tensors`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"


def _my_pad(
    tokenizer,
    inputs: Union[Dict[str, EncodedInput], BatchEncoding],
    max_length: Optional[int] = None,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    pad_to_multiple_of: Optional[int] = None,
    return_attention_mask: Optional[bool] = None,
    ) -> dict:
    # Load from model defaults
    if return_attention_mask is None:
        return_attention_mask = "attention_mask" in tokenizer.model_input_names
    required_input = inputs[tokenizer.model_input_names[0]]
    inputs['cate_ids'][0], inputs['cate_ids'][-1] = 0, 0
    for i, cate_idx in enumerate(inputs['cate_ids']):
        if cate_idx < 0 and i > 0:
            inputs['cate_ids'][i] = inputs['cate_ids'][i-1]
    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = len(required_input)
    if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
    if needs_to_be_padded:
        difference = max_length - len(required_input)
        if tokenizer.padding_side == "right":
            if return_attention_mask:
                inputs["attention_mask"] = [1] * len(required_input) + [0] * difference
            if "token_type_ids" in inputs:
                inputs["token_type_ids"] = (
                        inputs["token_type_ids"] + [tokenizer.pad_token_type_id] * difference
                )
            if "cate_ids" in inputs:
                inputs["cate_ids"] = (
                        inputs["cate_ids"] + [tokenizer.pad_token_type_id] * difference
                )
            if "special_tokens_mask" in inputs:
                inputs["special_tokens_mask"] = inputs["special_tokens_mask"] + [1] * difference
            inputs[tokenizer.model_input_names[0]] = required_input + [tokenizer.pad_token_id] * difference
        elif tokenizer.padding_side == "left":
            if return_attention_mask:
                inputs["attention_mask"] = [0] * difference + [1] * len(required_input)
            if "token_type_ids" in inputs:
                inputs["token_type_ids"] = [tokenizer.pad_token_type_id] * difference + inputs[
                    "token_type_ids"
                ]
            if "cate_ids" in inputs:
                inputs["cate_ids"] = (
                        inputs["cate_ids"] + [tokenizer.pad_token_type_id] * difference
                )
            if "special_tokens_mask" in inputs:
                inputs["special_tokens_mask"] = [1] * difference + inputs["special_tokens_mask"]
            inputs[tokenizer.model_input_names[0]] = [tokenizer.pad_token_id] * difference + required_input
        else:
            raise ValueError("Invalid padding strategy:" + str(tokenizer.padding_side))
    elif return_attention_mask and "attention_mask" not in inputs:
        inputs["attention_mask"] = [1] * len(required_input)
    return inputs


def _ym_pad_for_event_gen(
    tokenizer,
    inputs: Union[Dict[str, EncodedInput], BatchEncoding],
    max_length_in: Optional[int] = None,
    max_length_out: Optional[int] = None,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    pad_to_multiple_of: Optional[int] = None,
    return_attention_mask: Optional[bool] = None,
    ) -> dict:
    # Load from model defaults
    if return_attention_mask is None:
        return_attention_mask = "attention_mask" in tokenizer.model_input_names
    input_idx = inputs['input_ids']
    label_idx = inputs['label_ids']
    inputs['cate_ids'][0], inputs['cate_ids'][-1] = 0, 0
    inputs['label_ids'][0], inputs['label_ids'][-1] = -1, -1
    for i, cate_idx in enumerate(inputs['cate_ids']):
        if cate_idx < 0 and i > 0:
            inputs['cate_ids'][i] = inputs['cate_ids'][i-1]
    # if padding_strategy == PaddingStrategy.LONGEST:
    #     max_length_in = len(input_ids)
    #     max_length_out = len(label_ids)
    # if max_length_in is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
    #     max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(input_idx) != max_length_in
    difference_in = max_length_in - len(input_idx)
    difference_out = max_length_out - len(label_idx)
    # if tokenizer.padding_side == "right":
    assert tokenizer.padding_side == 'right'
    if return_attention_mask:
        inputs["attention_mask"] = [1] * len(input_idx) + [0] * difference_in
    if "token_type_ids" in inputs:
        inputs["token_type_ids"] = (
                inputs["token_type_ids"] + [tokenizer.pad_token_type_id] * difference_in
        )
    if "cate_ids" in inputs:
        inputs["cate_ids"] = (
                inputs["cate_ids"] + [tokenizer.pad_token_type_id] * difference_in
        )
    if "special_tokens_mask" in inputs:
        inputs["special_tokens_mask"] = inputs["special_tokens_mask"] + [1] * difference_in
    if 'label_ids' in inputs:
        inputs["label_ids"] = inputs["label_ids"] + [-1] * difference_out
    inputs['input_ids'] = input_idx + [tokenizer.pad_token_id] * difference_in
    return inputs


def _my_pad_for_event_gen(
    tokenizer,
    inputs: Union[Dict[str, EncodedInput], BatchEncoding],
    max_length_in: Optional[int] = None,
    max_length_out: Optional[int] = None,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    pad_to_multiple_of: Optional[int] = None,
    return_attention_mask: Optional[bool] = None,
    ) -> dict:
    # Load from model defaults
    if return_attention_mask is None:
        return_attention_mask = "attention_mask" in tokenizer.model_input_names
    input_idx = inputs['input_ids']
    label_idx = inputs['label_ids']
    inputs['cate_ids'][0], inputs['cate_ids'][-1] = 0, 0
    inputs['label_ids'][0], inputs['label_ids'][-1] = 0, 0
    for i, cate_idx in enumerate(inputs['cate_ids']):
        if cate_idx < 0 and i > 0:
            inputs['cate_ids'][i] = inputs['cate_ids'][i-1]
    # if padding_strategy == PaddingStrategy.LONGEST:
    #     max_length_in = len(input_ids)
    #     max_length_out = len(label_ids)
    # if max_length_in is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
    #     max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(input_idx) != max_length_in
    difference_in = max_length_in - len(input_idx)
    difference_out = max_length_out - len(label_idx)
    # if tokenizer.padding_side == "right":
    assert tokenizer.padding_side == 'right'
    if return_attention_mask:
        inputs["attention_mask"] = [1] * len(input_idx) + [0] * difference_in
    if "token_type_ids" in inputs:
        inputs["token_type_ids"] = (
                inputs["token_type_ids"] + [tokenizer.pad_token_type_id] * difference_in
        )
    if "cate_ids" in inputs:
        inputs["cate_ids"] = (
                inputs["cate_ids"] + [tokenizer.pad_token_type_id] * difference_in
        )
    if "special_tokens_mask" in inputs:
        inputs["special_tokens_mask"] = inputs["special_tokens_mask"] + [1] * difference_in
    if 'label_ids' in inputs:
        inputs["label_ids"] = inputs["label_ids"] + [0] * difference_out
    inputs['input_ids'] = input_idx + [tokenizer.pad_token_id] * difference_in
    return inputs


def my_pad(
        tokenizer,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, str]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
    if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
        encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

    # The model's main input name, usually `input_ids`, has be passed for padding
    if tokenizer.model_input_names[0] not in encoded_inputs:
        raise ValueError(
            "You should supply an encoding or a list of encodings to this method"
            f"that includes {tokenizer.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
        )

    required_input = encoded_inputs[tokenizer.model_input_names[0]]

    if not required_input:
        if return_attention_mask:
            encoded_inputs["attention_mask"] = []
        return encoded_inputs

    # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
    # and rebuild them afterwards if no return_tensors is specified
    # Note that we lose the specific device the tensor may be on for PyTorch

    first_element = required_input[0]
    # if isinstance(first_element, (list, tuple)):
    #     # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
    #     index = 0
    #     while len(required_input[index]) == 0:
    #         index += 1
    #     if index < len(required_input):
    #         first_element = required_input[index][0]
    # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
    # if not isinstance(first_element, (int, list, tuple)):
    #     if is_tf_available() and _is_tensorflow(first_element):
    #         return_tensors = "tf" if return_tensors is None else return_tensors
    #     elif is_torch_available() and _is_torch(first_element):
    #         return_tensors = "pt" if return_tensors is None else return_tensors
    #     elif isinstance(first_element, np.ndarray):
    #         return_tensors = "np" if return_tensors is None else return_tensors
    #     else:
    #         raise ValueError(
    #             f"type of {first_element} unknown: {type(first_element)}. "
    #             f"Should be one of a python, numpy, pytorch or tensorflow object."
    #         )
    #     for key, value in encoded_inputs.items():
    #         encoded_inputs[key] = to_py_obj(value)

    # Convert padding_strategy in PaddingStrategy
    padding_strategy, _, max_length, _ = tokenizer._get_padding_truncation_strategies(
        padding=padding, max_length=max_length, verbose=verbose
    )

    required_input = encoded_inputs[tokenizer.model_input_names[0]]
    # if required_input and not isinstance(required_input[0], (list, tuple)):
    #     encoded_inputs = _my_pad(
    #         tokenizer,
    #         encoded_inputs,
    #         max_length=max_length,
    #         padding_strategy=padding_strategy,
    #         pad_to_multiple_of=pad_to_multiple_of,
    #         return_attention_mask=return_attention_mask,
    #     )
    #     return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

    batch_size = len(required_input)
    assert all(
        len(v) == batch_size for v in encoded_inputs.values()
    ), "Some items in the output dictionary have a different batch size than others."

    # if padding_strategy == PaddingStrategy.LONGEST:
    max_length = max(len(inputs) for inputs in required_input)
    padding_strategy = PaddingStrategy.MAX_LENGTH

    batch_outputs = {}
    for i in range(batch_size):
        inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
        outputs = _my_pad(
            tokenizer,
            inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)

    return BatchEncoding(batch_outputs, tensor_type=return_tensors)


def ym_pad_for_event_gen(
        tokenizer,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, str]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
    if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
        encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

    # The model's main input name, usually `input_ids`, has be passed for padding
    if tokenizer.model_input_names[0] not in encoded_inputs:
        raise ValueError(
            "You should supply an encoding or a list of encodings to this method"
            f"that includes {tokenizer.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
        )
    input_ids = encoded_inputs[tokenizer.model_input_names[0]]
    # encoded_inputs['label_ids'] = encoded_inputs.pop('labels')
    # del encoded_inputs['labels']
    label_ids = encoded_inputs['label_ids']

    if not input_ids:
        if return_attention_mask:
            encoded_inputs["attention_mask"] = []
        return encoded_inputs

    # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
    # and rebuild them afterwards if no return_tensors is specified
    # Note that we lose the specific device the tensor may be on for PyTorch

    first_element = input_ids[0]
    # Convert padding_strategy in PaddingStrategy
    padding_strategy, _, max_length, _ = tokenizer._get_padding_truncation_strategies(
        padding=padding, max_length=max_length, verbose=verbose
    )

    batch_size = len(input_ids)
    assert all(
        len(v) == batch_size for v in encoded_inputs.values()
    ), "Some items in the output dictionary have a different batch size than others."

    # if padding_strategy == PaddingStrategy.LONGEST:
    max_length_input = max(len(inputs) for inputs in input_ids)
    max_length_output = max(len(labels) for labels in label_ids)
    padding_strategy = PaddingStrategy.MAX_LENGTH

    batch_outputs = {}
    for i in range(batch_size):
        inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
        outputs = _ym_pad_for_event_gen(
            tokenizer,
            inputs,
            max_length_in=max_length_input,
            max_length_out=max_length_output,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)

    return BatchEncoding(batch_outputs, tensor_type=return_tensors)


def my_pad_for_event_gen(
        tokenizer,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, str]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
    if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
        encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

    # The model's main input name, usually `input_ids`, has be passed for padding
    if tokenizer.model_input_names[0] not in encoded_inputs:
        raise ValueError(
            "You should supply an encoding or a list of encodings to this method"
            f"that includes {tokenizer.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
        )
    input_ids = encoded_inputs[tokenizer.model_input_names[0]]
    label_ids = encoded_inputs['label_ids']

    if not input_ids:
        if return_attention_mask:
            encoded_inputs["attention_mask"] = []
        return encoded_inputs

    # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
    # and rebuild them afterwards if no return_tensors is specified
    # Note that we lose the specific device the tensor may be on for PyTorch

    first_element = input_ids[0]
    # Convert padding_strategy in PaddingStrategy
    padding_strategy, _, max_length, _ = tokenizer._get_padding_truncation_strategies(
        padding=padding, max_length=max_length, verbose=verbose
    )

    batch_size = len(input_ids)
    assert all(
        len(v) == batch_size for v in encoded_inputs.values()
    ), "Some items in the output dictionary have a different batch size than others."

    # if padding_strategy == PaddingStrategy.LONGEST:
    max_length_input = max(len(inputs) for inputs in input_ids)
    max_length_output = max(len(labels) for labels in label_ids)
    padding_strategy = PaddingStrategy.MAX_LENGTH

    batch_outputs = {}
    for i in range(batch_size):
        inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
        outputs = _my_pad_for_event_gen(
            tokenizer,
            inputs,
            max_length_in=max_length_input,
            max_length_out=max_length_output,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)

    return BatchEncoding(batch_outputs, tensor_type=return_tensors)


class DataCollatorForEventPredYM:

    def __init__(self,
                 tokenizer=None,
                 padding=True,
                 max_length=None,
                 pad_to_multiple_of=None,
                 label_pad_token_id=-100
                 ):
        self.tokenizer=tokenizer
        self.padding=padding
        self.max_length=max_length
        self.pad_to_multiple_of=pad_to_multiple_of
        self.label_pad_token_id=label_pad_token_id
    # tokenizer: PreTrainedTokenizerBase
    # # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    # label_pad_token_id: int = -100

    def __call__(self, features):
        batch = ym_pad_for_event_gen(
            tokenizer=self.tokenizer,
            encoded_inputs=features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )
        # batch["labels"] = batch['label_ids']
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


class DataCollatorForEventGen:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
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
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    def __init__(self,
                 tokenizer=None,
                 padding=True,
                 max_length=None,
                 pad_to_multiple_of=None,
                 label_pad_token_id=-100
                 ):
        self.tokenizer=tokenizer
        self.padding=padding
        self.max_length=max_length
        self.pad_to_multiple_of=pad_to_multiple_of
        self.label_pad_token_id=label_pad_token_id
    # tokenizer: PreTrainedTokenizerBase
    # # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    # label_pad_token_id: int = -100

    def __call__(self, features):
        batch = my_pad_for_event_gen(
            tokenizer=self.tokenizer,
            encoded_inputs=features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors=None,
        )
        # batch["labels"] = batch['label_ids']
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch size, dec_hid_dim]
        # encoder_outputs: [batch size, src_len, enc_hid_dim]
        # batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src_len, dec_hid_dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)
        # attention= [batch size, src len]
        return F.softmax(attention, dim=1)


class AttDecoder(nn.Module):
    def __init__(self, output_vocab_size, dec_emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(output_vocab_size, dec_emb_dim)
        self.rnn = nn.GRU(enc_hid_dim + dec_emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + dec_emb_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.output_vocab_size = output_vocab_size
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def forward(self, dec_input, hidden, encoder_outputs):
        # dec_input = [batch_size]
        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [batch_size, src_len, enc_hid_dim]
        dec_input = dec_input.unsqueeze(0)
        # input = [1 (i.e. input_length), batch_size]
        embedded = self.dropout(self.embedding(dec_input))
        # embedded = [1, batch_size, emb_dim]
        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src_len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch_size, 1, enc_hid_dim]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc_hid_dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, enc_hid_dim + emb_dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch size, output dim]
        return prediction, hidden.squeeze(0)


class EventGenerator(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.trg_vocab_size = self.decoder.output_vocab_size

    def forward(self, input_ids, cate_ids, attention_mask, token_type_ids, label_ids, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = input_ids.shape[0]
        trg_len = label_ids.shape[1]
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, self.trg_vocab_size).to(self.device)
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs_ = self.encoder(input_ids, cate_ids, attention_mask, token_type_ids)
        encoder_outputs, hidden = encoder_outputs_[0], encoder_outputs_[1]
        # first input to the decoder is the <sos> tokens
        dec_input = label_ids[:, 0]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, hidden = self.decoder(dec_input, hidden, encoder_outputs)
            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = dec_output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            dec_input = label_ids[:, t] if teacher_force else top1
        return outputs

    def generate(self, input_ids, cate_ids, attention_mask, token_type_ids, info):
        preds = []
        assert 'idx2cui' in info and 'cui2idx' in info
        idx2cui, cui2idx = info['idx2cui'], info['cui2idx']
        with torch.no_grad():
            for i, (input_ids_, cate_ids_, attention_mask_, token_type_ids_) in \
                    enumerate(zip(input_ids, cate_ids, attention_mask, token_type_ids)):
                input_ids_ = input_ids_.reshape(1, -1)
                cate_ids_ = cate_ids_.reshape(1, -1)
                attention_mask_ = attention_mask_.reshape(1, -1)
                token_type_ids_ = token_type_ids_.reshape(1, -1)
                pred = []
                encoder_outputs_ = self.encoder(input_ids_, cate_ids_, attention_mask_, token_type_ids_)
                encoder_outputs, hidden = encoder_outputs_[0], encoder_outputs_[1]
                dec_input = torch.Tensor([cui2idx['<SOE>']]).int().to(self.device)
                continue_predict=True
                while continue_predict and len(pred) < info['max_len']:
                    dec_output, hidden = self.decoder(dec_input, hidden, encoder_outputs)
                    output = dec_output.cpu().numpy()[0]
                    idx2value = {idx: v for idx, v in enumerate(output)}
                    ranks = [idx for idx, v in sorted(idx2value.items(), key=lambda item: item[1], reverse=True)]
                    for idx in ranks:
                        if idx == cui2idx['<EOE>']:
                            continue_predict = False
                            break
                        elif idx not in pred:
                            pred.append(idx)
                            break
                preds.append(pred)
        return preds


class DataCollatorForTokenClassificationPlus:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
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
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    def __init__(self,
                 tokenizer=None,
                 padding=True,
                 max_length=None,
                 pad_to_multiple_of=None,
                 label_pad_token_id=-100
                 ):
        self.tokenizer=tokenizer
        self.padding=padding
        self.max_length=max_length
        self.pad_to_multiple_of=pad_to_multiple_of
        self.label_pad_token_id=label_pad_token_id
    # tokenizer: PreTrainedTokenizerBase
    # # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    # label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = my_pad(
            tokenizer=self.tokenizer,
            encoded_inputs=features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch













