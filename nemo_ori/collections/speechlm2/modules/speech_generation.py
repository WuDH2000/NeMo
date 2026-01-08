# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
import torchaudio

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.tts.modules import transformer_2501
from nemo.core.classes.module import NeuralModule
from nemo.collections.speechlm2.parts.precision import fp32_precision

def build_vocabs(subword_vocab_items):
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
    # subword_vocab_items = tokenizer.vocab.items()
    org_char_vocab = {subword: subword_id for subword, subword_id in subword_vocab_items if len(subword) == 1}
    sorted_char_vocab = dict(sorted(org_char_vocab.items(), key=lambda x: x[1]))
    char_vocab = {k: i for i, (k, _) in enumerate(sorted_char_vocab.items())}
    assert sorted(char_vocab.values()) == list(range(len(char_vocab)))
    subword_id_to_char_ids = {
        subword_id: tuple(char_vocab[char] for char in subword) for subword, subword_id in subword_vocab_items
    }

    assert max(subword_id_to_char_ids) == len(subword_id_to_char_ids) - 1
    # add padding_idx
    subword_padding_idx = len(subword_id_to_char_ids)
    subword_id_to_char_ids[subword_padding_idx] = (len(char_vocab),)

    return subword_id_to_char_ids, char_vocab, subword_padding_idx

    
def sequence_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    if max_length is None:
        max_length = int(lengths.max().item())
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


from nemo.core.classes.module import NeuralModule
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
import torch.nn.functional as F


class CharAwareSubwordEncoder(NeuralModule):
    def __init__(self, params: DictConfig, llm_tokenizer_vocab_items: dict = None):
        super().__init__()
        self.subword_id_to_char_ids, self.char_vocab, self.subword_padding_idx = build_vocabs(llm_tokenizer_vocab_items)
        self.embed_tokens = nn.Embedding(self.vocab_size + 1, params["d_model"], padding_idx=self.vocab_size)
        self.encoder = transformer_2501.Transformer(**params)

    @property
    def vocab_size(self):
        return len(self.char_vocab)

    def prepare_inputs(self, subword_ids: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        device = subword_ids.device

        subword_id_list = torch.masked_select(subword_ids, padding_mask).cpu().tolist()
        char_id_list = [list(self.subword_id_to_char_ids[x]) for x in subword_id_list]

        char_lengths = torch.tensor([len(x) for x in char_id_list], dtype=torch.long, device=device)
        batch_size = char_lengths.size(0)

        char_ids = torch.full((batch_size, int(char_lengths.max().item())), self.vocab_size, dtype=torch.long)
        for i in range(batch_size):
            char_ids[i, : char_lengths[i]] = torch.tensor(char_id_list[i])
        char_ids = char_ids.to(device=device)
        return char_ids, char_lengths

    def forward(self, subword_ids: Tensor, subword_mask: Tensor | None = None) -> Tensor:
        device = subword_ids.device
        if subword_mask is None:
            subword_mask = torch.ones_like(subword_ids).bool()
        else:
            subword_mask = subword_mask.bool()

        if subword_mask.ndim == 3:
            subword_mask = subword_mask.squeeze(-1)

        char_ids, char_lengths = self.prepare_inputs(subword_ids, subword_mask)

        char_mask = sequence_mask(char_lengths)
        char_emb = self.embed_tokens(char_ids)

        # char emb has the shape  [B*T, N, channels], where N is the max number of chars tokens decoded from bpe tokens
        x = self.encoder(
            x=char_emb,
            x_mask=char_mask
        )['output']

        # Get average embedding over the chars
        mean_emb = ((x / char_mask.unsqueeze(-1).sum(1, keepdim=True)) * char_mask.unsqueeze(-1)).sum(1)
        subword_emb = torch.zeros((subword_mask.size(0), subword_mask.size(1), mean_emb.size(-1)), device=device)
        subword_emb[subword_mask.unsqueeze(-1).expand(-1, -1, mean_emb.size(-1))] = mean_emb.view(-1)

        return subword_emb


class SemanticTokenPredictor(NeuralModule):
    """
    MLP-based Semantic Token Predictor (Legacy/Simple version)
    
    This is a lightweight MLP that predicts semantic tokens frame-by-frame
    without considering temporal dependencies between semantic tokens.
    
    Design:
    1. Single-layer semantic tokens (WhisperVQ, 12.5 FPS)
    2. Sequential prediction: text_token first, then use text_token to predict semantic_token
    3. Input: [llm_hidden, text_embed] -> Output: semantic_logits
    4. No speaker_emb needed (semantic is speaker-independent)
    5. Lightweight design
    """
    
    def __init__(
        self,
        llm_hidden_dim: int,
        semantic_vocab_size: int = 16384,  # WhisperVQ vocabulary size
        hidden_dim: int = 512,  # Internal hidden dimension
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.llm_hidden_dim = llm_hidden_dim
        self.semantic_vocab_size = semantic_vocab_size
        self.hidden_dim = hidden_dim
        
        # Fusion layer: fuse [llm_hidden, text_embed]
        # Input: (B, T, llm_hidden_dim * 2) -> Output: (B, T, hidden_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(llm_hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Prediction head: predict semantic token
        # Input: (B, T, hidden_dim) -> Output: (B, T, semantic_vocab_size)
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, semantic_vocab_size),
        )
    
    def forward(
        self,
        llm_hidden: torch.Tensor,     # (B, T, llm_hidden_dim) - LLM's last_hidden_state
        text_embeds: torch.Tensor,    # (B, T, llm_hidden_dim) - embed_tokens(text_tokens)
        semantic_labels: torch.Tensor = None,  # (B, T) - Ground truth semantic tokens (training only)
        loss_mask: torch.Tensor = None,        # (B, T) - Valid positions for loss
    ):
        """
        Forward pass for semantic token prediction.
        
        Args:
            llm_hidden: (B, T, D) LLM's last_hidden_state containing context and semantic info
            text_embeds: (B, T, D) Embeddings of predicted text tokens
            semantic_labels: (B, T) Ground truth semantic tokens (training only)
            loss_mask: (B, T) Valid positions mask (True=valid, False=padding)
        
        Returns:
            dict with:
                - semantic_logits: (B, T, semantic_vocab_size)
                - semantic_loss: scalar (if semantic_labels provided)
        """
        B, T, D = llm_hidden.shape
        
        # Step 1: Concatenate llm_hidden and text_embeds
        # Shape: (B, T, 2*D)
        fused_input = torch.cat([llm_hidden, text_embeds], dim=-1)
        
        # Step 2: Fusion
        # Shape: (B, T, hidden_dim)
        fused = self.fusion_layer(fused_input)
        
        # Step 3: Predict semantic tokens
        # Shape: (B, T, semantic_vocab_size)
        semantic_logits = self.semantic_head(fused)
        
        result = {"semantic_logits": semantic_logits}
        
        # Step 4: Calculate loss if training
        if semantic_labels is not None:
            if loss_mask is not None:
                # Only compute loss at valid positions
                semantic_loss = F.cross_entropy(
                    semantic_logits[loss_mask],
                    semantic_labels[loss_mask],
                    reduction='mean'
                )
            else:
                # Compute loss at all positions
                semantic_loss = F.cross_entropy(
                    semantic_logits.reshape(-1, self.semantic_vocab_size),
                    semantic_labels.reshape(-1),
                    reduction='mean'
                )
            result["semantic_loss"] = semantic_loss
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        llm_hidden: torch.Tensor,     # (B, 1, llm_hidden_dim) - Single timestep
        text_embed: torch.Tensor,     # (B, 1, llm_hidden_dim) - Single timestep text embedding
        temperature: float = 0.9,
        topk: int = 20,
        past_semantic_tokens: torch.Tensor = None,  # Unused, for API compatibility
    ):
        """
        Generate semantic token for one timestep (inference).
        
        Args:
            llm_hidden: (B, 1, D) Current timestep's LLM hidden state
            text_embed: (B, 1, D) Current timestep's predicted text token embedding
            temperature: Sampling temperature
            topk: Top-k sampling
            past_semantic_tokens: Unused, for API compatibility with TransformerSemanticPredictor
        
        Returns:
            semantic_token: (B,) Sampled semantic token
        """
        B = llm_hidden.size(0)
        
        # Concatenate and fuse
        fused_input = torch.cat([llm_hidden, text_embed], dim=-1)  # (B, 1, 2*D)
        fused = self.fusion_layer(fused_input)  # (B, 1, hidden_dim)
        semantic_logits = self.semantic_head(fused)  # (B, 1, semantic_vocab_size)
        
        # Remove time dimension: (B, semantic_vocab_size)
        semantic_logits = semantic_logits.squeeze(1)
        
        # Sample with temperature and top-k
        semantic_token = self._sample_topk(semantic_logits, topk, temperature)
        
        return semantic_token
    
    def _sample_topk(self, logits, k, temperature):
        """
        Top-k sampling with temperature.
        
        Args:
            logits: (B, vocab_size)
            k: top-k value
            temperature: temperature for sampling
        
        Returns:
            sampled_tokens: (B,)
        """
        # Apply temperature
        logits = logits / temperature
        
        # Get top-k values and threshold
        topk_values = torch.topk(logits, k, dim=-1)[0]
        threshold = topk_values[:, -1:]
        
        # Mask logits below threshold
        indices_to_remove = logits < threshold
        logits[indices_to_remove] = -float('Inf')
        
        # Softmax and sample
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return sampled_tokens
    
    def reset_cache(self):
        """No-op for MLP predictor, for API compatibility."""
        pass


class TransformerSemanticPredictor(NeuralModule):
    """
    Transformer-based Semantic Token Predictor with autoregressive modeling.
    
    Design principles:
    1. Leverage semantic token history via causal self-attention
    2. Use cross-attention to attend to condition (llm_hidden + text_embeds)
    3. Support KV cache for efficient autoregressive inference
    
    Architecture:
    - Condition Encoder: Fuses llm_hidden and text_embeds, optional self-attention
    - Semantic Decoder: Causal Transformer decoder with cross-attention to condition
    
    Input:
    - llm_hidden: (B, T, D_llm) - LLM's hidden states
    - text_embeds: (B, T, D_llm) - Predicted text token embeddings
    - semantic_tokens: (B, T) - Shifted semantic tokens (training) or history (inference)
    
    Output:
    - semantic_logits: (B, T, semantic_vocab_size)
    """
    
    def __init__(
        self,
        llm_hidden_dim: int,
        semantic_vocab_size: int = 16384,
        d_model: int = 1024,           # Transformer internal dimension
        n_heads: int = 8,
        n_layers: int = 4,             # Decoder layers
        n_cond_layers: int = 2,        # Condition encoder layers
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        
        self.llm_hidden_dim = llm_hidden_dim
        self.semantic_vocab_size = semantic_vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # ============ Condition Encoder ============
        # Fuse llm_hidden and text_embeds to create cross-attention memory
        self.condition_proj = nn.Linear(llm_hidden_dim * 2, d_model)
        self.condition_norm = nn.LayerNorm(d_model)
        
        # Optional: self-attention layers for condition enhancement
        if n_cond_layers > 0:
            cond_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-norm for stability
            )
            self.condition_encoder = nn.TransformerEncoder(cond_layer, num_layers=n_cond_layers)
        else:
            self.condition_encoder = None
        
        # ============ Semantic Token Embeddings ============
        # +1 for BOS token
        self.semantic_bos_id = semantic_vocab_size
        self.semantic_embedding = nn.Embedding(semantic_vocab_size + 1, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        
        # ============ Transformer Decoder ============
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # ============ Output Head ============
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, semantic_vocab_size)
        
        # ============ Cache for Inference ============
        self._use_cache = False
        self._condition_cache = None
        self._semantic_cache = None
        self._cache_pos = 0
        
        # Pre-compute causal mask
        self.register_buffer("causal_mask", None, persistent=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights similar to GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask (True = masked position)."""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def _prepare_condition(
        self,
        llm_hidden: torch.Tensor,      # (B, T, D_llm)
        text_embeds: torch.Tensor,     # (B, T, D_llm)
    ) -> torch.Tensor:
        """
        Prepare condition (memory for cross-attention).
        
        Returns:
            condition: (B, T, d_model)
        """
        # Concatenate and project
        fused = torch.cat([llm_hidden, text_embeds], dim=-1)  # (B, T, 2*D_llm)
        condition = self.condition_proj(fused)  # (B, T, d_model)
        condition = self.condition_norm(condition)
        
        # Optional: encode with self-attention
        if self.condition_encoder is not None:
            condition = self.condition_encoder(condition)
        
        return condition
    
    def _prepare_semantic_input(
        self,
        semantic_tokens: torch.Tensor,  # (B, T)
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Prepare semantic decoder input.
        
        Args:
            semantic_tokens: (B, T) - Shifted semantic tokens
            start_pos: Starting position for positional encoding (for cached inference)
        
        Returns:
            semantic_embeds: (B, T, d_model)
        """
        B, T = semantic_tokens.shape
        device = semantic_tokens.device
        
        # Embed semantic tokens
        semantic_embeds = self.semantic_embedding(semantic_tokens)  # (B, T, d_model)
        
        # Add positional encoding
        positions = torch.arange(start_pos, start_pos + T, device=device).unsqueeze(0).expand(B, -1)
        pos_embeds = self.pos_encoding(positions)  # (B, T, d_model)
        
        semantic_embeds = semantic_embeds + pos_embeds
        
        return semantic_embeds
    
    def forward(
        self,
        llm_hidden: torch.Tensor,              # (B, T, D_llm)
        text_embeds: torch.Tensor,             # (B, T, D_llm)
        semantic_labels: torch.Tensor = None,  # (B, T) - ground truth for loss
        loss_mask: torch.Tensor = None,        # (B, T) - valid positions
    ) -> dict:
        """
        Training forward pass with teacher forcing.
        
        Input/Target alignment:
        - Input:  [BOS, s_0, s_1, ..., s_{T-2}]  (shifted right)
        - Target: [s_0, s_1, s_2, ..., s_{T-1}]  (original labels)
        
        Args:
            llm_hidden: (B, T, D_llm) LLM hidden states
            text_embeds: (B, T, D_llm) Text token embeddings
            semantic_labels: (B, T) Ground truth semantic tokens
            loss_mask: (B, T) Valid positions mask
        
        Returns:
            dict with semantic_logits and optionally semantic_loss
        """
        B, T, _ = llm_hidden.shape
        device = llm_hidden.device
        
        # ========== Step 1: Prepare Condition (Cross-Attention Memory) ==========
        condition = self._prepare_condition(llm_hidden, text_embeds)  # (B, T, d_model)
        
        # ========== Step 2: Prepare Semantic Input (Shifted) ==========
        if semantic_labels is not None:
            # Training: shift labels right, prepend BOS
            bos_tokens = torch.full((B, 1), self.semantic_bos_id, device=device, dtype=torch.long)
            shifted_semantic = torch.cat([bos_tokens, semantic_labels[:, :-1]], dim=1)  # (B, T)
        else:
            # No labels: use all BOS (not useful, but for API consistency)
            shifted_semantic = torch.full((B, T), self.semantic_bos_id, device=device, dtype=torch.long)
        
        semantic_input = self._prepare_semantic_input(shifted_semantic)  # (B, T, d_model)
        
        # ========== Step 3: Transformer Decoder ==========
        causal_mask = self._get_causal_mask(T, device)
        
        decoder_output = self.decoder(
            tgt=semantic_input,           # (B, T, d_model) - semantic sequence
            memory=condition,             # (B, T, d_model) - condition from llm+text
            tgt_mask=causal_mask,         # Causal mask for autoregressive
            tgt_is_causal=True,
        )  # (B, T, d_model)
        
        # ========== Step 4: Output Projection ==========
        decoder_output = self.output_norm(decoder_output)
        semantic_logits = self.output_proj(decoder_output)  # (B, T, semantic_vocab_size)
        
        result = {"semantic_logits": semantic_logits}
        
        # ========== Step 5: Calculate Loss ==========
        if semantic_labels is not None:
            if loss_mask is not None:
                loss = F.cross_entropy(
                    semantic_logits[loss_mask],
                    semantic_labels[loss_mask],
                    reduction='mean'
                )
            else:
                loss = F.cross_entropy(
                    semantic_logits.reshape(-1, self.semantic_vocab_size),
                    semantic_labels.reshape(-1),
                    reduction='mean'
                )
            result["semantic_loss"] = loss
        
        return result
    
    def reset_cache(self):
        """Reset KV cache for new sequence."""
        self._use_cache = False
        self._condition_cache = None
        self._semantic_cache = None
        self._cache_pos = 0
    
    @torch.no_grad()
    def generate(
        self,
        llm_hidden: torch.Tensor,          # (B, T_cond, D_llm) - Full or incremental
        text_embed: torch.Tensor,          # (B, T_cond, D_llm) - Full or incremental
        temperature: float = 0.9,
        topk: int = 20,
        past_semantic_tokens: torch.Tensor = None,  # (B, T_past) - Previously generated tokens
    ) -> torch.Tensor:
        """
        Generate next semantic token autoregressively.
        
        This method supports two modes:
        1. Full sequence mode: Pass full llm_hidden/text_embed with past_semantic_tokens
        2. Incremental mode: Use reset_cache() then call generate() step by step
        
        Args:
            llm_hidden: (B, T, D_llm) LLM hidden states (full or current step)
            text_embed: (B, T, D_llm) Text embeddings (full or current step)
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            past_semantic_tokens: (B, T_past) Previously generated semantic tokens
        
        Returns:
            next_token: (B,) Next semantic token
        """
        B = llm_hidden.size(0)
        device = llm_hidden.device
        
        # Prepare full condition sequence
        condition = self._prepare_condition(llm_hidden, text_embed)  # (B, T_cond, d_model)
        
        # Prepare semantic input: [BOS] + past_tokens
        if past_semantic_tokens is None or past_semantic_tokens.size(1) == 0:
            # First step: only BOS
            semantic_input_ids = torch.full((B, 1), self.semantic_bos_id, device=device, dtype=torch.long)
        else:
            # Prepend BOS to past tokens
            bos_tokens = torch.full((B, 1), self.semantic_bos_id, device=device, dtype=torch.long)
            semantic_input_ids = torch.cat([bos_tokens, past_semantic_tokens], dim=1)  # (B, T_past+1)
        
        T_dec = semantic_input_ids.size(1)
        semantic_input = self._prepare_semantic_input(semantic_input_ids)  # (B, T_dec, d_model)
        
        # Causal mask for decoder
        causal_mask = self._get_causal_mask(T_dec, device)
        
        # Decode
        decoder_output = self.decoder(
            tgt=semantic_input,
            memory=condition,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
        )  # (B, T_dec, d_model)
        
        # Get last position logits
        last_hidden = self.output_norm(decoder_output[:, -1, :])  # (B, d_model)
        logits = self.output_proj(last_hidden)  # (B, semantic_vocab_size)
        
        # Sample with temperature and top-k
        next_token = self._sample_topk(logits, topk, temperature)
        
        return next_token
    
    def _sample_topk(self, logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
        """
        Top-k sampling with temperature.
        
        Args:
            logits: (B, vocab_size)
            k: top-k value
            temperature: temperature for sampling
        
        Returns:
            sampled_tokens: (B,)
        """
        # Apply temperature
        logits = logits / temperature
        
        # Get top-k values and threshold
        topk_values = torch.topk(logits, k, dim=-1)[0]
        threshold = topk_values[:, -1:]
        
        # Mask logits below threshold
        logits = logits.clone()
        logits[logits < threshold] = -float('Inf')
        
        # Softmax and sample
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return sampled_tokens


class TransformerARSpeechDecoder(NeuralModule):
    def __init__(
        self,
        speech_decoder_parms: DictConfig,
        lantent_dim: int,
        num_audio_codebooks: int,
        num_audio_tokens_per_codebook: int,
        llm_tokenizer_vocab_items: dict,
    ):
        super().__init__()
        self.use_input_cache = False
        self.speech_decoder_parms = speech_decoder_parms
        self.lantent_dim = lantent_dim
        self.num_audio_codebooks = num_audio_codebooks
        self.num_audio_tokens_per_codebook = num_audio_tokens_per_codebook
        # optional configs
        self.cfg_unconditional_prob = self.speech_decoder_parms.pop("cfg_unconditional_prob", None)
        self.cfg_scale = self.speech_decoder_parms.pop("cfg_scale", 2.5)
        self.cond_on_prev_audio_tokens = self.speech_decoder_parms.pop("cond_on_prev_audio_tokens", True)
        self.detach_input = self.speech_decoder_parms.pop("detach_input", False)
        self.cond_on_text_tokens = self.speech_decoder_parms.pop("cond_on_text_tokens", False)
        self.cond_on_llm_latent = self.speech_decoder_parms.pop("cond_on_llm_latent", True)
        self.cond_on_asr_emb = self.speech_decoder_parms.pop("cond_on_asr_emb", False)
        self.drop_asr_emb_prob = self.speech_decoder_parms.pop("drop_asr_emb_prob", 0.0)
        self.asr_emb_dim = self.speech_decoder_parms.pop("asr_emb_dim", 512) 
        self.cond_on_modality_adapter_emb = self.speech_decoder_parms.pop("cond_on_modality_adapter_emb", False)
        self.modality_adapter_emb_quantizer_levels = self.speech_decoder_parms.pop("modality_adapter_emb_quantizer_levels", None)
        self.use_speaker_encoder = self.speech_decoder_parms.pop("use_speaker_encoder", True)
        self.speaker_embedding_dim = self.speech_decoder_parms.pop("speaker_embedding_dim", 192)
        self.inference_speaker_reference = self.speech_decoder_parms.pop("inference_speaker_reference", None)
        self.max_speaker_reference_len = self.speech_decoder_parms.pop("max_speaker_reference_len", 5)
        self.speaker_encoder_model_name = self.speech_decoder_parms.pop("speaker_encoder_model_name", 'titanet_large')
        self.cond_on_char_embedding = self.speech_decoder_parms.pop("cond_on_char_embedding", True)
        self.cas_n_layers = self.speech_decoder_parms.pop("cas_n_layers", 1)
        self.use_cas_cache = self.speech_decoder_parms.pop("use_cas_cache", True)
        self.use_gated_fusion = self.speech_decoder_parms.pop("use_gated_fusion", False)

        if self.use_speaker_encoder:
            # load speaker encoder
            self.setup_speaker_encoder()
            # speaker encoder projection
            self.speaker_encoder_emb_projection = nn.Linear(self.speaker_embedding_dim, self.speech_decoder_parms["d_model"])
            if self.use_gated_fusion:
                self.se_gate_proj = nn.Linear(2 * self.speech_decoder_parms["d_model"], 1)


            # generate a random speaker embedding
            inference_speaker_embedding = torch.randn([1, 1, self.speaker_embedding_dim])
            self.register_buffer("inference_speaker_embedding", inference_speaker_embedding)
            # if inference_speaker_reference is provided, replace random embedding by the reference speaker embedding
            if self.inference_speaker_reference:
                self.update_inference_speaker_embedding(self.inference_speaker_reference)

        if not self.cond_on_llm_latent and not self.cond_on_text_tokens and not self.cond_on_char_embedding:
            raise ValueError(
                "At least one of 'cond_on_text_tokens' or 'cond_on_llm_latent' or 'cond_on_char_embedding' must be True for the Speech Decoder."
            )

        if self.cond_on_text_tokens and self.cond_on_char_embedding:
            raise ValueError(
                "'cond_on_text_tokens' and 'cond_on_char_embedding' are incompatible, you need to select one of these options !!"
            )

        # projection to adapt llm embeddings into the same shape of speech decoder expected input
        if self.cond_on_llm_latent and lantent_dim != self.speech_decoder_parms["d_model"]:
            self.input_proj = nn.Linear(lantent_dim, self.speech_decoder_parms["d_model"])
        else:
            self.input_proj = None

        # instanciate T5-TTS decoder to full compatibility and potentialy load pretrained model
        self.t5_decoder = transformer_2501.Transformer(**self.speech_decoder_parms)

        # projection to predict audio codes
        self.final_proj = nn.Linear(self.speech_decoder_parms["d_model"], num_audio_codebooks * num_audio_tokens_per_codebook)

        if self.cond_on_char_embedding:
            self.cas_params = dict(self.speech_decoder_parms)
            # uses only 1 layer for the char encoder
            self.cas_params["n_layers"] = self.cas_n_layers
            self.cas_encoder = CharAwareSubwordEncoder(self.cas_params, llm_tokenizer_vocab_items)
            if self.use_gated_fusion and self.cond_on_llm_latent:
                self.char_gate_proj = nn.Linear(2 * self.speech_decoder_parms["d_model"], 1)

        # create embeddings for encode input tokens
        if self.cond_on_prev_audio_tokens:
            audio_embeddings = []
            for _ in range(self.num_audio_codebooks):
                audio_embeddings.append(nn.Embedding(num_audio_tokens_per_codebook, self.speech_decoder_parms["d_model"]))

            self.audio_embeddings = nn.ModuleList(audio_embeddings)
            if self.use_gated_fusion:
                self.audio_gate_proj = nn.Linear(2 * self.speech_decoder_parms["d_model"], 1)

        if self.cond_on_text_tokens:
            self.text_embeddings = nn.Embedding(len(llm_tokenizer_vocab_items), self.speech_decoder_parms["d_model"])
            if self.use_gated_fusion and self.cond_on_llm_latent:
                self.text_gate_proj = nn.Linear(2 * self.speech_decoder_parms["d_model"], 1)

        # if cond on llm latent create the projection to sum the embeddings
        if self.cond_on_llm_latent and (self.cond_on_text_tokens or self.cond_on_char_embedding):
            self.text_input_projection = nn.Linear(self.speech_decoder_parms["d_model"], self.speech_decoder_parms["d_model"])

        if self.cond_on_modality_adapter_emb:
            if self.modality_adapter_emb_quantizer_levels:
                from nemo.collections.tts.modules.audio_codec_modules import FiniteScalarQuantizer
                bottleneck_dim = len(self.modality_adapter_emb_quantizer_levels)
                self.modality_adapter_emb_quantizer_projection = nn.Linear(lantent_dim, bottleneck_dim)
                self.mae_vector_quantizer = FiniteScalarQuantizer(self.modality_adapter_emb_quantizer_levels)
                self.modality_adapter_emb_projection = nn.Linear(bottleneck_dim, self.speech_decoder_parms["d_model"])
            else:
                self.modality_adapter_emb_projection = nn.Linear(lantent_dim, self.speech_decoder_parms["d_model"])

        if self.cond_on_asr_emb:
            self.asr_emb_projection = nn.Linear(self.asr_emb_dim, self.speech_decoder_parms["d_model"])

    def setup_speaker_encoder(self):
        with fp32_precision():
            self.speaker_encoder = EncDecSpeakerLabelModel.from_pretrained(model_name=self.speaker_encoder_model_name)

        # freeze the pretrained speaker encoder
        self.speaker_encoder.eval()
        self.speaker_encoder.freeze()

        for p in self.speaker_encoder.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def update_inference_speaker_embedding(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        audio_len = torch.tensor([audio.size(1)]).long()
        self.inference_speaker_embedding = self.get_speaker_embedding(audio.to(self.device), audio_len.to(self.device), sr)

    def get_speaker_embedding(self, audio, audio_len, sr):
        # limit max audio len to avoid memory waste
        audio = audio[:, : int(self.max_speaker_reference_len*sr)]
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            with torch.no_grad():
                model_sr = self.speaker_encoder._cfg.train_ds.get('sample_rate', 16000)
                audio_resampled = torchaudio.functional.resample(audio, sr, model_sr)
                audio_len_resampled = audio_len * (model_sr / sr )
                _, g = self.speaker_encoder(input_signal=audio_resampled, input_signal_length=audio_len_resampled.long())
                g = g.unsqueeze(1)
        return g.to(self.inference_speaker_embedding.dtype)


    def forward(self, hidden_states, speech_mask, input_audio_tokens=None, target_text_tokens=None, modality_adapter_emb=None, asr_emb=None, speaker_encoder_emb=None, temperature=0.7, topk=80, greedy=True):
        # LLM returns T, B, F so reshape it
        if hidden_states is not None:
            hidden_states = hidden_states.transpose(0, 1).contiguous() # .reshape(B, T, F) # from [T, B, F] to [B, T, F]
    
            if self.detach_input:
                hidden_states = hidden_states.detach()

        # input cache needed due our transformer kv cache implementation expect the whole left context
        if self.use_input_cache:
            if self.cache["hidden_states"] is None:
                self.cache["hidden_states"] = hidden_states
            else:
                self.cache["hidden_states"] = torch.cat([self.cache["hidden_states"], hidden_states], dim=1)
                hidden_states = self.cache["hidden_states"]

            if self.cache["speech_mask"] is None:
                self.cache["speech_mask"] = speech_mask
            else:
                self.cache["speech_mask"] = torch.cat([self.cache["speech_mask"], speech_mask], dim=1)
                speech_mask = self.cache["speech_mask"]

            if self.cache["input_audio_tokens"] is None:
                self.cache["input_audio_tokens"] = input_audio_tokens
            else:
                self.cache["input_audio_tokens"] = torch.cat([self.cache["input_audio_tokens"], input_audio_tokens], dim=1)
                input_audio_tokens = self.cache["input_audio_tokens"]

            if self.cache["target_text_tokens"] is None:
                self.cache["target_text_tokens"] = target_text_tokens
            else:
                if target_text_tokens is not None:
                    self.cache["target_text_tokens"] = torch.cat([self.cache["target_text_tokens"], target_text_tokens], dim=1)
                    target_text_tokens = self.cache["target_text_tokens"]

        # map hidden states to the shape of the
        if hidden_states is not None and self.input_proj is not None:
            speech_decoder_input = self.input_proj(hidden_states)
        else:
            speech_decoder_input = hidden_states

        # workaround for inference, because during inference speech_mask will be None
        if speech_mask is None:
            speech_mask = torch.ones(
                (speech_decoder_input.size(0), speech_decoder_input.size(1)),
                device=speech_decoder_input.device,
            )

        # if cond on text tokens, sum text tokens with the llm latent
        if self.cond_on_text_tokens and target_text_tokens is not None:
            text_tokens_embedded = self.text_embeddings(target_text_tokens)

            # if cond_on_llm_latent use a projection to sum the embeddings
            if self.cond_on_llm_latent:
                speech_decoder_input = self.text_input_projection(speech_decoder_input)
                if self.use_gated_fusion:
                    gate_input = torch.cat([speech_decoder_input, text_tokens_embedded], dim=-1)
                    gate = torch.sigmoid(self.text_gate_proj(gate_input))
                    speech_decoder_input = gate * speech_decoder_input + (1 - gate) * text_tokens_embedded
                else:
                    speech_decoder_input = speech_decoder_input + text_tokens_embedded
            else:
                speech_decoder_input = text_tokens_embedded
        elif self.cond_on_text_tokens and target_text_tokens is None:
            raise ValueError(
                "target_text_tokens should not be None !"
            )

        if self.cond_on_char_embedding:
            # if inference time cache char_embs to speedup inference
            if self.use_input_cache and self.use_cas_cache:
                char_emb_last = self.cas_encoder(target_text_tokens[:, -1:], subword_mask=speech_mask[:, -1:])
                if self.cache["char_embs"] is None:
                    self.cache["char_embs"] = char_emb_last
                else:
                    self.cache["char_embs"] = torch.cat([self.cache["char_embs"], char_emb_last], dim=1)
                char_embs = self.cache["char_embs"]
            else:
                char_embs = self.cas_encoder(target_text_tokens, subword_mask=speech_mask)

            # if cond_on_llm_latent use a projection to sum the embeddings
            if self.cond_on_llm_latent:
                speech_decoder_input = self.text_input_projection(speech_decoder_input)
                if self.use_gated_fusion:
                    gate_input = torch.cat([speech_decoder_input, char_embs], dim=-1)
                    gate = torch.sigmoid(self.char_gate_proj(gate_input))
                    speech_decoder_input = gate * speech_decoder_input + (1 - gate) * char_embs
                else:
                    speech_decoder_input = speech_decoder_input + char_embs
            else:
                speech_decoder_input = char_embs

        if self.cond_on_modality_adapter_emb:
            if self.detach_input:
                modality_adapter_emb = modality_adapter_emb.detach()

            if self.use_input_cache:
                if self.modality_adapter_emb_quantizer_levels:
                    modality_adapter_emb = self.modality_adapter_emb_quantizer_projection(modality_adapter_emb[:, -1:])
                    modality_adapter_emb, _ = self.mae_vector_quantizer(inputs=modality_adapter_emb.transpose(1, 2), input_len=None)
                    modality_adapter_emb = modality_adapter_emb.transpose(1, 2)

                if self.cache["modality_adapter_emb"] is None:
                    self.cache["modality_adapter_emb"] = modality_adapter_emb
                else:
                    if modality_adapter_emb is not None:
                        self.cache["modality_adapter_emb"] = torch.cat([self.cache["modality_adapter_emb"], modality_adapter_emb], dim=1)
                        modality_adapter_emb = self.cache["modality_adapter_emb"]
            else:
                if self.modality_adapter_emb_quantizer_levels:
                    modality_adapter_emb = self.modality_adapter_emb_quantizer_projection(modality_adapter_emb)
                    modality_adapter_emb, _ = self.mae_vector_quantizer(inputs=modality_adapter_emb.transpose(1, 2), input_len=None)
                    modality_adapter_emb = modality_adapter_emb.transpose(1, 2)

            modality_adapter_emb = self.modality_adapter_emb_projection(modality_adapter_emb)
            speech_decoder_input = speech_decoder_input + modality_adapter_emb

        if self.cond_on_asr_emb:
            if self.detach_input:
                asr_emb = asr_emb.detach()

            if self.use_input_cache:
                if self.cache["asr_emb"] is None:
                    self.cache["asr_emb"] = asr_emb
                else:
                    if asr_emb is not None:
                        self.cache["asr_emb"] = torch.cat([self.cache["asr_emb"], asr_emb], dim=1)
                        asr_emb = self.cache["asr_emb"]

            asr_emb = self.asr_emb_projection(asr_emb)
            if self.training and self.drop_asr_emb_prob:
                if torch.rand(1).item() < self.drop_asr_emb_prob:
                    asr_emb = torch.zeros_like(asr_emb)

            speech_decoder_input = speech_decoder_input + asr_emb

        if self.use_speaker_encoder:
            # for inference uses the inference cached speaker embedding
            # ToDo: replace the repeat operation by adding over all inference time steps to speedup
            if self.use_input_cache and not self.training:
                speaker_encoder_emb = self.inference_speaker_embedding
                # repeat speaker encoder embedding to match the time and batch dimention
                speaker_encoder_emb = speaker_encoder_emb.repeat(speech_decoder_input.size(0), speech_decoder_input.size(1), 1)
            else:
                # repeat speaker encoder embedding to match the time dimention
                if speaker_encoder_emb.size(1) != speech_decoder_input.size(1):
                    speaker_encoder_emb = speaker_encoder_emb.repeat(1, speech_decoder_input.size(1), 1)

            speaker_encoder_emb = self.speaker_encoder_emb_projection(speaker_encoder_emb)

            if self.use_gated_fusion:
                gate_input = torch.cat([speech_decoder_input, speaker_encoder_emb], dim=-1)
                gate = torch.sigmoid(self.se_gate_proj(gate_input))
                speech_decoder_input = gate * speech_decoder_input + (1 - gate) * speaker_encoder_emb
            else:
                speech_decoder_input = speech_decoder_input + speaker_encoder_emb

        if self.cfg_unconditional_prob:
            if self.training:
                # if training drop the "text" conditioning in a percentage of batch
                if torch.rand(1).item() < self.cfg_unconditional_prob:
                    # make the whole batch zeros to the unconditional model
                    # ToDo: move it to cache to need to just create a 1 frame tensor in inference
                    speech_decoder_input = torch.zeros_like(speech_decoder_input)
            else:
                # if inference or evaluation create a zero tensor for speech decoder input and concatenate it to compute unconditional logits
                speech_decoder_input_zeros = torch.zeros_like(speech_decoder_input)
                speech_decoder_input = torch.cat([speech_decoder_input, speech_decoder_input_zeros], dim=0)
                # duplicate mask to match the new shape
                speech_mask = torch.cat([speech_mask, speech_mask], dim=0)
                # if cond on prev tokens enabled, so duplicate the tokens to the new shape
                if self.cond_on_prev_audio_tokens:
                    input_audio_tokens = torch.cat([input_audio_tokens, input_audio_tokens], dim=0)

        # audio tokens should not be dropped by cfg, so we keep it here
        if self.cond_on_prev_audio_tokens:
            if self.detach_input:
                input_audio_tokens = input_audio_tokens.detach()

            audio_tokens_embedded = self.embed_audio_tokens(
                input_audio_tokens.transpose(1, 2).contiguous()
            )  # (B, T', E)

            if self.use_gated_fusion:
                gate_input = torch.cat([speech_decoder_input, audio_tokens_embedded], dim=-1)
                gate = torch.sigmoid(self.audio_gate_proj(gate_input))
                speech_decoder_input = gate * speech_decoder_input + (1 - gate) * audio_tokens_embedded
            else:
                speech_decoder_input = speech_decoder_input + audio_tokens_embedded

        decoder_out = self.t5_decoder(x=speech_decoder_input, x_mask=speech_mask)['output']

        # if it is true we need to return just the last autoregressive step, it is valid because for 1 frame input we produce 1 frame ouput
        if self.use_input_cache:
            decoder_out = decoder_out[:, -1:, :]

        # get the logits of all codebooks
        all_code_logits = self.final_proj(decoder_out)

        # if using cfg and it is in inference or evaluation mix unconditional and coditional logits
        if self.cfg_unconditional_prob and not self.training:
            batch_size = all_code_logits.size(0) // 2
            cond_logits = all_code_logits[:batch_size]
            uncond_logits = all_code_logits[batch_size:]
            all_code_logits = (1 - self.cfg_scale) * uncond_logits + self.cfg_scale * cond_logits

        # sample for inference
        if self.use_input_cache and not self.training:
            sampled_audio_tokens = self.sample_codes_from_logits(all_code_logits, temperature=temperature, topk=topk, greedy=greedy)
        else:
            sampled_audio_tokens = None

        return all_code_logits, sampled_audio_tokens

    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.7, topk=80, greedy=True):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = []
        for idx in range(self.num_audio_codebooks):
            si = idx * self.num_audio_tokens_per_codebook
            ei = si + self.num_audio_tokens_per_codebook
            codebook_logits = all_code_logits_t[:, :, si:ei] # (B, num_tokens_per_codebook)
            B, T = codebook_logits.size(0), codebook_logits.size(1)
            if greedy:
                codebook_preds = torch.argmax(codebook_logits, dim=-1)
            else:
                codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0] # (B, topk)
                codebook_probs = torch.softmax(codebook_logits_topk / temperature, dim=-1) # (B, num_tokens_per_codebook)
                codebook_preds = torch.multinomial(codebook_probs, 1) # (B, 1)
            all_preds.append(codebook_preds.view(B, T).contiguous().long()) # T, B to be compatible with megatron

        return all_preds

    def embed_audio_tokens(self, audio_tokens):
        # Add and average the embeddings of the audio tokens across the codebooks
        audio_embedding = None
        for c in range(self.num_audio_codebooks):
            embedding = self.audio_embeddings[c](audio_tokens[:, c, :])
            if audio_embedding is None:
                audio_embedding = embedding
            else:
                audio_embedding = audio_embedding + embedding
        audio_embedding = audio_embedding / audio_tokens.size(1)
        return audio_embedding

    def reset_input_and_kv_cache(self, use_cache):
        if use_cache:
            print("Enabling input and KV cache!")
        else:
            print("Disabling input and KV cache!")

        self.use_input_cache = use_cache
        self.cache = self._init_cache()
        self.t5_decoder.reset_cache(use_cache=use_cache)

    @staticmethod
    def _init_cache():
        return {
            'hidden_states': None,
            'speech_mask': None,
            'input_audio_tokens': None,
            'target_text_tokens': None,
            'modality_adapter_emb': None,
            'asr_emb': None,
            'char_embs': None,
        }
