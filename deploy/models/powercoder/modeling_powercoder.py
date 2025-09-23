
from typing import Callable, Optional, Union

import torch
from torch import nn

from retention.triton import power_retention, power_retention_inference

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from .configuration_powercoder import PowerCoderConfig
from .kvgs_dynamic_cache import Cache, DynamicCache

class PowerCoderMLP(nn.Module):
    def __init__(self, config: PowerCoderConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, config.intermediate_size, bias=config.use_bias)
        self.c_proj = nn.Linear(config.intermediate_size, embed_dim, bias=config.use_bias)
        self.act = ACT2FN[config.hidden_act]
        self.residual_dropout = config.residual_dropout

    def forward(self, hidden_states: Optional[tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.residual_dropout, training=self.training)
        return hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_power_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = 2*torch.log(torch.abs( torch.matmul(query, key_states.transpose(2, 3)) * scaling + 1e-5))
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class PowerCoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PowerCoderConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.g_proj = nn.Linear(config.hidden_size, config.num_key_value_heads,                 bias=config.use_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.use_bias)
        self.residual_dropout = config.residual_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        padding_starts: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        chunk_size: Optional[int] = None,
        switch_over_seq_len: Optional[int] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        interpolate_exp_amount = kwargs.get('interpolate_exp', 0)
        assert 0 <= interpolate_exp_amount <= 1, f'{interpolate_exp_amount=}'
        run_exp = interpolate_exp_amount > 0

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        gate_states = self.g_proj(hidden_states).view(hidden_shape[:-1]).transpose(1, 2)
        gate_states = nn.functional.logsigmoid(gate_states.to(torch.float32))

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states, gate_states, state, sum_of_keys = past_key_value.update_kv(key_states, value_states, gate_states, self.layer_idx, cache_kwargs)

        if run_exp:
            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

            exp_attn_output, exp_attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                is_causal=True,
                attention_mask=None,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        if query_states.shape[2] == 1:
            key_len = key_states.shape[2]
            power_attn_output, state, sum_of_keys = power_retention_inference(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                gate_states.transpose(1, 2),
                initial_state=state,
                sum_of_keys=sum_of_keys,
                deg=2,
                scale=self.scaling,
                switch_over_seq_len=switch_over_seq_len,
            )
            if switch_over_seq_len is not None and key_len >= switch_over_seq_len:
                past_key_value.clean_kv(self.layer_idx)
                past_key_value.update_state(state, sum_of_keys, self.layer_idx, cache_kwargs)

        else:
            key_len = key_states.shape[2]
            power_attn_output = power_retention(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                gate_states.transpose(1, 2),
                deg=2,
                scale=self.scaling,
                chunk_size=chunk_size, # enable chunked prefilling by default
                switch_over_seq_len=switch_over_seq_len,
            )

        if interpolate_exp_amount == 1:
            attn_output = exp_attn_output
        elif interpolate_exp_amount == 0:
            attn_output = power_attn_output
        else:
            attn_output = interpolate_exp_amount * exp_attn_output + (1 - interpolate_exp_amount) * power_attn_output

        assert attn_output.shape == (input_shape[0], query_states.shape[2], self.config.num_attention_heads, self.head_dim),\
            f'{attn_output.shape=} {(input_shape[0], query_states.shape[2], self.config.num_attention_heads, self.head_dim)=}'
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = nn.functional.dropout(
            attn_output, p=self.residual_dropout, training=self.training
        )  # diff with Llama

        return attn_output


class PowerCoderDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PowerCoderConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PowerCoderAttention(config=config, layer_idx=layer_idx)
        self.mlp = PowerCoderMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_starts: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        chunk_size: Optional[int] = None,
        switch_over_seq_len: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            padding_starts=padding_starts,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            chunk_size=chunk_size,
            switch_over_seq_len=switch_over_seq_len,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class PowerCoderRotaryEmbedding(nn.Module):
    def __init__(self, config: PowerCoderConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class PowerCoderPreTrainedModel(PreTrainedModel):
    config: PowerCoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PowerCoderDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": PowerCoderDecoderLayer,
        "attentions": PowerCoderAttention,
    }


@auto_docstring
class PowerCoderModel(PowerCoderPreTrainedModel):
    def __init__(self, config: PowerCoderConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [PowerCoderDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.rotary_emb = PowerCoderRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.embedding_dropout = config.embedding_dropout

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        chunk_size: Optional[int] = None,
        switch_over_seq_len: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Always use our local DynamicCache implementation for compatibility with gating
        if use_cache:
            if past_key_values is None or not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        # causal_mask = mask_function(
        #     config=self.config,
        #     input_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     cache_position=cache_position,
        #     past_key_values=past_key_values,
        #     position_ids=position_ids,
        # )
        padding_starts = attention_mask.argmin(-1) if attention_mask is not None else None

        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.embedding_dropout, training=self.training
        )  # main diff with Llama

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                padding_starts=padding_starts,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                chunk_size=chunk_size,
                switch_over_seq_len=switch_over_seq_len,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class PowerCoderForCausalLM(PowerCoderPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = PowerCoderModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        chunk_size: Optional[int] = None,
        switch_over_seq_len: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, PowerCoderForCausalLM

        >>> model = PowerCoderForCausalLM.from_pretrained("meta-PowerCoder/PowerCoder-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-PowerCoder/PowerCoder-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```

        Args:
            input_ids (`Optional[torch.LongTensor]`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`Optional[torch.Tensor]`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`Optional[torch.LongTensor]`, *optional*):
                Indices of positions of each input sequence tokens.
            past_key_values (`Optional[Cache]`, *optional*):
                Cache containing pre-computed key and value states for attention layers, used for faster inference.
                If `use_cache` is True, the cache will be used and updated with new key/value states.
            inputs_embeds (`Optional[torch.FloatTensor]`, *optional*):
                Pre-computed input embeddings. Useful for scenarios where you want to compute embeddings separately.
            labels (`Optional[torch.LongTensor]`, *optional*):
                Labels for computing language modeling loss.
            use_cache (`Optional[bool]`, *optional*):
                If True, past key/value states are returned and can be used for future predictions.
            cache_position (`Optional[torch.LongTensor]`, *optional*):
                Position indices for cached key/value states when using incremental decoding.
            logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to 0):
                Number of logits to compute from the end of the sequence, or specific indices to compute.
            chunk_size (`Optional[int]`, *optional*):
                Chunk size for training and prefilling.
            switch_over_seq_len (`Optional[int]`, *optional*):
                Sequence length threshold for state update.
            **kwargs:
                Additional arguments passed to the underlying model's forward method.

        Returns:
            `CausalLMOutputWithPast`: A dataclass containing:
                - loss (`Optional[torch.FloatTensor]`): Language modeling loss if labels were provided.
                - logits (`torch.FloatTensor`): Prediction scores for the vocabulary.
                - past_key_values (`Optional[Cache]`): Updated key/value states for attention layers if use_cache=True.
                - hidden_states (`Optional[Tuple[torch.FloatTensor]]`): Model's hidden states.
                - attentions (`Optional[Tuple[torch.FloatTensor]]`): Attention weights if output_attentions=True.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            chunk_size=chunk_size,
            switch_over_seq_len=switch_over_seq_len,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PowerCoderForSequenceClassification(GenericForSequenceClassification, PowerCoderPreTrainedModel):
    pass


class PowerCoderForTokenClassification(GenericForTokenClassification, PowerCoderPreTrainedModel):
    pass


__all__ = [
    "PowerCoderForCausalLM",
    "PowerCoderModel",
    "PowerCoderPreTrainedModel",
    "PowerCoderForSequenceClassification",
    "PowerCoderForTokenClassification",
]
