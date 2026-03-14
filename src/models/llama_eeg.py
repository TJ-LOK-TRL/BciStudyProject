"""
Modified LLaMA implementation for EEG — based on lma.py from EEGEncoder paper.
Key differences from HuggingFace standard:
  - All Linear layers replaced with LinearL2 (manual L2 loss)
  - Dropout added to attention weights
  - Dropout added inside MLP (before gate activation)
  - dropout_ratio and weight_decay propagated via LlamaConfig
"""
import math
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel


# ─── L2-regularised layers ───────────────────────────────────────────────────

class LinearL2(nn.Module):
    """Linear layer that exposes l2_loss() for manual regularisation."""

    def __init__(self, in_features: int, out_features: int, weight_decay: float = 0.0, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def l2_loss(self) -> torch.Tensor:
        return self.weight_decay * torch.sum(self.linear.weight ** 2)


# ─── Positional encoding ─────────────────────────────────────────────────────

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
    sin = sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _make_causal_mask(
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    bsz, tgt_len = shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    cond = torch.arange(tgt_len, device=device)
    mask.masked_fill_(cond < (cond + 1).view(tgt_len, 1), 0)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    bsz, src_len = mask.size()
    tgt_len = tgt_len or src_len
    expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    return (1.0 - expanded).masked_fill((1.0 - expanded).bool(), torch.finfo(dtype).min)


# ─── Transformer blocks ───────────────────────────────────────────────────────

class EEGLlamaMLP(nn.Module):
    """SwiGLU MLP with dropout before gate — key difference from HuggingFace."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, dropout_ratio: float, weight_decay: float):
        super().__init__()
        self.gate_proj = LinearL2(hidden_size, intermediate_size, weight_decay=weight_decay)
        self.down_proj = LinearL2(intermediate_size, hidden_size, weight_decay=weight_decay)
        self.up_proj = LinearL2(hidden_size, intermediate_size, weight_decay=weight_decay)
        self.act_fn = ACT2FN[hidden_act]
        self.dp = nn.Dropout(dropout_ratio)   # <-- não existe no HuggingFace standard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dropout before gate activation — original lma.py behaviour
        return self.down_proj(self.dp(self.act_fn(self.gate_proj(x))) * self.up_proj(x))


class EEGLlamaAttention(nn.Module):
    """MHA with dropout on attention weights — key difference from HuggingFace."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        wd = config.weight_decay
        self.q_proj = LinearL2(self.hidden_size, self.hidden_size, weight_decay=wd)
        self.k_proj = LinearL2(self.hidden_size, self.hidden_size, weight_decay=wd)
        self.v_proj = LinearL2(self.hidden_size, self.hidden_size, weight_decay=wd)
        self.o_proj = LinearL2(self.hidden_size, self.hidden_size, weight_decay=wd)
        self.attention_drop = nn.Dropout(config.dropout_ratio)   # <-- não existe no HuggingFace standard
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, self.max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2] + (past_key_value[0].shape[-2] if past_key_value else 0)
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v) if use_cache else None

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = self.attention_drop(attn_weights)    # <-- dropout ANTES do softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value


class EEGLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = EEGLlamaAttention(config)
        self.mlp = EEGLlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            dropout_ratio=config.dropout_ratio,
            weight_decay=config.weight_decay,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple:
        # pre-norm + residual (same as original)
        residual = hidden_states
        hidden_states, attn_weights, present_kv = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_kv,)
        return outputs


# ─── Full model ───────────────────────────────────────────────────────────────

class EEGLlamaModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([EEGLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ) -> Optional[torch.Tensor]:
        combined = None
        if input_shape[-1] > 1:
            combined = _make_causal_mask(input_shape, inputs_embeds.dtype, inputs_embeds.device, past_key_values_length)
        if attention_mask is not None:
            expanded = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined = expanded if combined is None else expanded + combined
        return combined

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len, _ = inputs_embeds.shape
        past_kv_length = past_key_values[0][0].shape[2] if past_key_values else 0

        if position_ids is None:
            position_ids = torch.arange(past_kv_length, seq_len + past_kv_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq_len + past_kv_length), device=inputs_embeds.device)

        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (bsz, seq_len), inputs_embeds, past_kv_length)

        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            layer_out = layer(hidden_states, attention_mask, position_ids, past_kv, output_attentions, use_cache)
            hidden_states = layer_out[0]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class EEGLlamaForCausalLM(PreTrainedModel):
    """
    LLaMA adapted for EEG — use this instead of HuggingFace LlamaForCausalLM.
    All LinearL2 layers expose l2_loss() for manual regularisation in training loop.
    """
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = EEGLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return CausalLMOutputWithPast(
            logits=self.lm_head(outputs.last_hidden_state),
            hidden_states=outputs.hidden_states,
        )