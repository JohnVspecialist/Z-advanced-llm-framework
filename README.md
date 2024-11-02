import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union, Callable

@dataclass
class ModelConfig:
    """Enhanced model configuration with smart defaults and validation"""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    head_dim: int = 128
    max_position_embeddings: int = 4096
    rope_theta: float = 10000
    rope_scaling: Optional[float] = None
    sliding_window: int = 4096
    gradient_checkpointing: bool = True
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    compile: bool = True
    use_flash_attention: bool = True
    use_paged_attention: bool = True
    activation_function: str = "silu"
    tie_word_embeddings: bool = True
    use_cache: bool = True
    def __post_init__(self):
        """Validate and adjust configuration parameters"""
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads cannot be greater than num_attention_heads")

        # Optimize model dimensions
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = self.intermediate_size - (self.intermediate_size % 256)

        # Adjust for hardware compatibility
        if self.device == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Enable gradient checkpointing for large models
        if self.hidden_size >= 2048:
            self.gradient_checkpointing = True

class RotaryEmbedding(nn.Module):
    """Enhanced Rotary Positional Embedding with caching and interpolation"""
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: int = 10000,
        scaling_factor: Optional[float] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
# Cache computations
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False
        )
        self._set_cos_sin_cache(max_position_embeddings, scaling_factor)

    def _set_cos_sin_cache(self, seq_len: int, scaling_factor: Optional[float] = None):
        self.max_seq_len_cached = seq_len
        if scaling_factor is not None:
            seq_len = int(seq_len * scaling_factor)
        
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.float32), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, self.scaling_factor)
            
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )
    class RMSNorm(nn.Module):
    """Optimized RMSNorm implementation"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).to(x.dtype)
        return output * self.weight

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Efficiently rotate half of the input features"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to queries and keys"""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(-2)
        sin = sin[position_ids].unsqueeze(-2)
    else:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
class PagedAttention(nn.Module):
    """Memory-efficient paged attention implementation"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        self.rope = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            scaling_factor=config.rope_scaling
        )

        # Initialize Q/K/V projections
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False
        )
        
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )
        
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False
        )

        self.attention_dropout = nn.Dropout(config.dropout)
        self.qk_bmm_workspace = None
        def _setup_paged_attention(self, batch_size: int, seq_len: int):
        """Initialize workspace for paged attention"""
        if (
            self.qk_bmm_workspace is None or
            self.qk_bmm_workspace.size(0) < batch_size or
            self.qk_bmm_workspace.size(1) < seq_len
        ):
            self.qk_bmm_workspace = torch.empty(
                (batch_size, seq_len, seq_len),
                device=self.q_proj.weight.device,
                dtype=torch.float32
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape
        head_dim = self.head_dim

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and apply rotary embeddings
        query_states = query_states.view(batch_size, seq_length, self.num_heads, head_dim)
        key_states = key_states.view(batch_size, seq_length, self.num_kv_heads, head_dim)
        value_states = value_states.view(batch_size, seq_length, self.num_kv_heads, head_dim)

        cos, sin = self.rope(query_states, seq_length)
        query_state
        # Handle past key-values
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
            
        past_key_value = (key_states, value_states) if self.config.use_cache else None

        # Efficient attention computation
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.config.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Scaled dot-product attention with memory-efficient implementation
            scale = 1.0 / math.sqrt(head_dim)
            attn_weights = torch.einsum('bhqd,bhkd->bhqk', query_states, key_states) * scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = self.attention_dropout(attn_weights)
            attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.view(batch_size, seq_length, self.num_heads * head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value, attn_weights if output_attentions else None
    class MLP(nn.Module):
    """Enhanced MLP with SwiGLU activation"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    """Enhanced Transformer block with improved architecture"""
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        self.self_attn = PagedAttention(config)
        self.mlp = MLP(config)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Progressive layer scaling
        self.layer_scale = math.exp(-2 * math.log(2 * config.num_hidden_layers) / config.num_hidden_layers * layer_idx)
        def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, past_key_value, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.layer_scale * hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.layer_scale * hidden_states

        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (past_key_value,)
            
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs

class KVCache:
    """Efficient key-value cache implementation"""
    def __init__(self, max_batch_size: int, max_seq_length: int, num_layers: int, num_heads: int, head_dim: int):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cache = {}
        def update(self, layer_idx: int, batch_idx: int, key: torch.Tensor, value: torch.Tensor):
        cache_key = (layer_idx, batch_idx)
        if cache_key not in self.cache:
            self.cache[cache_key] = [key, value]
        else:
            self.cache[cache_key][0] = torch.cat([self.cache[cache_key][0], key], dim=1)
            self.cache[cache_key][1] = torch.cat([self.cache[cache_key][1], value], dim=1)
            
        # Trim if exceeding max length
        if self.cache[cache_key][0].size(1) > self.max_seq_length:
            self.cache[cache_key][0] = self.cache[cache_key][0][:, -self.max_seq_length:]
            self.cache[cache_key][1] = self.cache[cache_key][1][:, -self.max_seq_length:]

    def get(self, layer_idx: int, batch_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        cache_key = (layer_idx, batch_idx)
        if cache_key in self.cache:
            return tuple(self.cache[cache_key])
        return None

    def reset(self):
        self.cache.clear()

class EnhancedLLM(nn.Module):
    """Enhanced Language Model with improved architecture and features"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Initialize weights
        self.apply(self._init_weights)

        # Setup KV cache
        self.kv_cache = KVCache(
            max_batch_size=32,  # Adjustable
            max_seq_length=config.max_position_embeddings,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_key_value_heads,
            head_dim=config.head_dim
        )

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.enable_gradient_checkpointing()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        for layer in self.layers:
            layer.gradient_checkpointing = True

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.embed_tokens = value
        def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)

        # Initialize outputs
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None
        # Process through layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache += (layer_outputs[1],)

            if output_attentions:
                all_self_attns += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (next_cache,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_self_attns,)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "loss": loss,
        }
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        min_length: int = 0,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # Initialize generation settings
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]

        # Reset KV cache
        self.kv_cache.reset()

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        while cur_len < max_length and unfinished_sequences.sum() > 0:
            # Forward pass through model
            outputs = self.forward(
                input_ids=input_ids[:, -1:],  # Only process last token
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=[self.kv_cache.get(i, 0) for i in range(self.config.num_hidden_layers)]
            )

            next_token_logits = F.linear(outputs["last_hidden_state"][:, -1], self.embed_tokens.weight)

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        next_token_logits[i, previous_token] /= repetition_penalty
                        # Filter tokens based on top-k and top-p
            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]

            # Update input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                attention_mask.new_ones((batch_size, 1))
            ], dim=-1)

            cur_len += 1

        return input_ids
    # Example usage
if __name__ == "__main__":
    # Initialize model
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
    )
    model = EnhancedLLM(config)

    # Example training data
    train_dataset = torch.utils.data.TensorDataset(
        torch.randint(0, config.vocab_size, (1000, 512)),
        torch.randint(0, config.vocab_size, (1000, 512)),
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
    )

    # Example usage for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training loop example
    model.train()
    for batch in train_dataloader:
        input_ids, labels = batch
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Example generation
    input_text = torch.randint(0, config.vocab_size, (1, 10))  # Example input
    generated = model.generate(
        input_text,
        max_length=50,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )
    
