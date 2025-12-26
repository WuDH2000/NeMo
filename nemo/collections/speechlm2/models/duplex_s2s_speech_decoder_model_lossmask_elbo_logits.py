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
import os
import random
import tempfile
import uuid
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor, nn, nonzero
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache, WhisperFeatureExtractor
import torch.distributed.checkpoint as dcp

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str, tokens_to_str_extract
from nemo.collections.speechlm2.modules.speech_generation import SemanticTokenPredictor, TransformerSemanticPredictor
from nemo.collections.speechlm2.modules.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from nemo.collections.speechlm2.modules.speech_tokenizer.utils import extract_speech_token
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TurnTakingMetrics
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


class GatedFusion(nn.Module):
    """
    Gated fusion module to dynamically balance text and audio embeddings.
    
    The gate is conditioned on token type (PAD vs non-PAD):
    - When predicting PAD (listening): audio_weight >> text_weight (since text_emb is repetitive)
    - When predicting actual text (speaking): balanced weights based on learned gate
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Gate network: learns to weight text importance based on text embedding
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        text_embeds: torch.Tensor, 
        audio_embeds: torch.Tensor,
        is_pad_mask: torch.Tensor = None  # (B, T), True where token is PAD
    ) -> torch.Tensor:
        """
        Args:
            text_embeds: (B, T, D)
            audio_embeds: (B, T, D)
            is_pad_mask: (B, T), optional, True where token is PAD
        
        Returns:
            fused_embeds: (B, T, D)
        """
        # Ensure input dtype consistency for mixed precision training
        input_dtype = text_embeds.dtype
        
        # Compute text importance based on text embedding
        text_gate = self.text_gate(text_embeds)  # (B, T, 1)
        audio_gate = 1.0 - text_gate
        
        # If PAD mask provided, strongly bias towards audio
        if is_pad_mask is not None:
            # When PAD: text_gate -> 0.1 (small), audio_gate -> 0.9 (large)
            pad_mask_expanded = is_pad_mask.unsqueeze(-1).to(dtype=input_dtype)  # (B, T, 1)
            text_gate = text_gate * (1.0 - pad_mask_expanded) + 0.1 * pad_mask_expanded
            audio_gate = 1.0 - text_gate
        
        fused = text_gate * text_embeds + audio_gate * audio_embeds
        
        return fused

class WhisperEncoderWrapper(nn.Module):
    """
    Wrapper for WhisperEncoder to support inputs_embeds and bypass conv layers.
    Also ensures compatibility with FSDP by keeping the forward pass within a Module.
    """
    def __init__(self, original_encoder, gradient_checkpointing=False):
        super().__init__()
        self.encoder = original_encoder
        self.config = original_encoder.config
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, inputs_embeds):
        # inputs_embeds: (B, T, D)
        hidden_states = inputs_embeds
        
        # Add positional embeddings
        # We explicitly call the embed_positions module to ensure FSDP compatibility (if needed)
        # WhisperPositionalEmbedding forward expects input_ids shape or position_ids
        seq_len = hidden_states.shape[1]
        
        # Defensive check: if sequence length exceeds max_source_positions, we clamp/interpolate?
        # User requested NO truncation. So we try to get embeddings for the full length.
        # If the underlying module is Sinusoidal, it might support arbitrary lengths or throw error.
        # If it's learned nn.Embedding, it will throw index error.
        # Whisper is typically Sinusoidal but stored as weights.
        
        # Hack: Generate position_ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0) # (1, seq_len)
        
        # Try to get pos embeddings. 
        # Note: If FSDP shards this, we rely on FSDP to gather.
        # However, for Sinusoidal with fixed weights, FSDP might not shard it.
        # Let's hope max_pos is sufficient or we handle it.
        
        # Check max length of embedding layer
        if hasattr(self.encoder.embed_positions, 'weight'):
             max_pos = self.encoder.embed_positions.weight.shape[0]
        elif hasattr(self.encoder.embed_positions, 'num_embeddings'):
             max_pos = self.encoder.embed_positions.num_embeddings
        else:
             max_pos = 1500 # Default fallback
             
        if seq_len <= max_pos:
             # Standard call
             # Note: Whisper's embed_positions might expect (input_ids) or (position_ids=...)
             # Looking at HF source: forward(input_ids=None, past_key_values_length=0, position_ids=None)
             # It uses input_ids.shape[1] + past to generate ids if position_ids is None.
             # So passing a dummy input_ids of correct shape is safest.
             dummy_ids = torch.zeros((1, seq_len), dtype=torch.long, device=hidden_states.device)
             pos_embeds = self.encoder.embed_positions(dummy_ids)
        else:
             # Interpolation strategy
             # Get all available positions
             dummy_ids = torch.zeros((1, max_pos), dtype=torch.long, device=hidden_states.device)
             all_pos_embeds = self.encoder.embed_positions(dummy_ids) # (1, max_pos, dim)
             
             # Interpolate
             pos_embeds = all_pos_embeds.transpose(1, 2)
             pos_embeds = torch.nn.functional.interpolate(pos_embeds, size=seq_len, mode='linear', align_corners=False)
             pos_embeds = pos_embeds.transpose(1, 2)
        
        hidden_states = hidden_states + pos_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.config.dropout, training=self.training)

        for layer in self.encoder.layers:
            if self.training and self.gradient_checkpointing:
                 def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, None, None)
                    return custom_forward
                 layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    use_reentrant=False
                 )
            else:
                layer_outputs = layer(hidden_states, None, None)
            hidden_states = layer_outputs[0]

        hidden_states = self.encoder.layer_norm(hidden_states)
        return hidden_states

class DuplexS2SSpeechDecoderModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # move back text channel by x, in inference it advance the text channel prediction by x frames
        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.

        # Load LLM first
        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()

        # Handle different model types with all their specific configurations
        if 'Nemotron' in self.cfg.pretrained_llm:
            # ====== NEMOTRON-SPECIFIC HANDLING ======
            # Tokenizer with override tokens from config
            # self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True, **self.cfg.get("override_tokens", {}))
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token = '<SPECIAL_12>'

            self.llm = getattr(llm, self.cfg.get("base_model_name", "backbone"))

            self.lm_head = llm.lm_head

            embed_tokens_name = self.cfg.get("embed_tokens_name", "embeddings")

            self.embed_tokens = getattr(self.llm, embed_tokens_name)

            delattr(self.llm, embed_tokens_name)

        elif 'Qwen2.5' in self.cfg.pretrained_llm:
            # ====== QWEN2.5-SPECIFIC HANDLING ======
            # Tokenizer with special token setup
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            # For Qwen, '<|im_start|>' is a common choice for a BOS token.
            # You can check your tokenizer's vocabulary for the best candidate.
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            # self.tokenizer.bos_token = '<|im_start|>'
            # self.tokenizer.eos_token = '<|im_end|>'
            special_tokens = {
                "bos_token": "<|im_start|>", 
                "eos_token": "<|im_end|>", 
                "additional_special_tokens": ["<|cot_start|>", "<|cot_end|>"]}
            if self.cfg.get("use_extra_id_for_pad", False):
                special_tokens["pad_token"] = '<|extra_1|>'
            try:
                num_added = self.tokenizer.add_special_tokens(special_tokens)
                print(self.tokenizer.tokenizer.all_special_tokens)
            except Exception as e:
                logging.warning(f"Failed adding CoT tokens to tokenizer: {e}")
                num_added = 0

            try:
                if num_added and num_added > 0:
                    target_model = llm.model if hasattr(llm, "model") else llm
                    if hasattr(target_model, "resize_token_embeddings"):
                        target_model.resize_token_embeddings(self.tokenizer.vocab_size)
            except Exception as e:
                logging.warning(f"Failed resizing token embeddings after adding CoT tokens: {e}")

            # Standard model access
            self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
            self.lm_head = llm.lm_head
            # Note: we have to "move out" the token embedding outside of LLM to avoid
            #       messing up FSDP/TP hooks.
            self.embed_tokens = self.llm.embed_tokens
            self.cotstart_id = self.tokenizer.tokens_to_ids("<|cot_start|>")
            self.cotend_id = self.tokenizer.tokens_to_ids("<|cot_end|>")
            print('cotstart_id', self.cotstart_id)
            print('cotend_id', self.cotend_id)
            print('self.tokenizer.tokenizer.all_special_tokens', self.tokenizer.tokenizer.all_special_tokens)
            del self.llm.embed_tokens

        else:

            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)

            # Standard model access
            self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
            self.lm_head = llm.lm_head
            # Note: we have to "move out" the token embedding outside of LLM to avoid
            #       messing up FSDP/TP hooks.
            self.embed_tokens = self.llm.embed_tokens
            del self.llm.embed_tokens

        maybe_install_lora(self)

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        setup_speech_encoder(self)

        # Initialize gated fusion module if enabled
        if self.cfg.get("use_gated_fusion", False):
            self.gated_fusion = GatedFusion(hidden_size=self.llm.config.hidden_size)
            # Match dtype with LLM for mixed precision training
            if hasattr(self.llm, 'dtype'):
                self.gated_fusion = self.gated_fusion.to(dtype=self.llm.dtype)
        else:
            self.gated_fusion = None

        # Setup semantic token generation components
        self._codebook_size = 16384
        self._num_codebooks = 1

        # WhisperVQ tokenizer for semantic token extraction (target only)
        self.whispervq = WhisperVQEncoder.from_pretrained(
            "THUDM/glm-4-voice-tokenizer",
            cache_dir='/mnt/donghang-jfs/pretrained_models',
        ).float().eval()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "THUDM/glm-4-voice-tokenizer",
            cache_dir='/mnt/donghang-jfs/pretrained_models',
        )

        # Semantic token predictor for predicting output semantic tokens
        # Supports two modes: MLP (lightweight) or Transformer (with temporal modeling)
        # Design: Sequential prediction (text first, then semantic)
        self._use_transformer_semantic = self.cfg.get("use_transformer_semantic_predictor", False)
        
        if self._use_transformer_semantic:
            self.semantic_predictor = TransformerSemanticPredictor(
                llm_hidden_dim=self.llm.config.hidden_size,
                semantic_vocab_size=self._codebook_size,  # 16384 - WhisperVQ semantic vocabulary
                d_model=self.cfg.get("semantic_predictor_d_model", 1024),
                n_heads=self.cfg.get("semantic_predictor_n_heads", 8),
                n_layers=self.cfg.get("semantic_predictor_n_layers", 4),
                n_cond_layers=self.cfg.get("semantic_predictor_n_cond_layers", 2),
                dim_feedforward=self.cfg.get("semantic_predictor_dim_feedforward", 2048),
                dropout=self.cfg.get("semantic_predictor_dropout", 0.1),
                max_seq_len=self.cfg.get("semantic_predictor_max_seq_len", 4096),
            )
        else:
            self.semantic_predictor = SemanticTokenPredictor(
                llm_hidden_dim=self.llm.config.hidden_size,
                semantic_vocab_size=self._codebook_size,  # 16384 - WhisperVQ semantic vocabulary
                hidden_dim=self.cfg.get("semantic_predictor_hidden_dim", 512),
                dropout=self.cfg.get("semantic_predictor_dropout", 0.1),
            )

        # Cached control codes for audio decoding (kept for potential future use)
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
            )

        if self.cfg.get("pretrained_s2s_model", None):
            self.init_from_model_from_ckpt(self.cfg.pretrained_s2s_model)
            # Ensure gated_fusion dtype matches after checkpoint loading
            if self.gated_fusion is not None and hasattr(self.llm, 'dtype'):
                self.gated_fusion = self.gated_fusion.to(dtype=self.llm.dtype)
        if self.cfg.get("pretrained_fsdp_s2s_model", None):
            self.init_from_fsdp_checkpoint_directory(self.cfg.pretrained_fsdp_s2s_model)
            print('init_from_model_from_fsdp_ckpt', self.cfg.get("pretrained_fsdp_s2s_model"))

        # Add binary classification head for COT/response classification (output dim=1 for sigmoid)
        # self.cot_classification_head = torch.nn.Linear(self.llm.config.hidden_size, 1)
        self.cot_classification_head = torch.nn.Sequential(
            torch.nn.Linear(self.llm.config.hidden_size, 2 * self.llm.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.llm.config.hidden_size, self.llm.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.llm.config.hidden_size, 1),
        )

        # Add non-causal TransformerEncoder for ELBO approach
        self.non_causal_encoder_type = self.cfg.get("non_causal_encoder_type", None)
        llm_hidden_size = self.llm.config.hidden_size
        self.non_causal_proj_in = None
        self.non_causal_proj_out = None

        if 'trans' in self.non_causal_encoder_type.lower():
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            encoder_layer = TransformerEncoderLayer(
                d_model=llm_hidden_size,
                nhead=8,  # Number of attention heads
                dim_feedforward=llm_hidden_size * 2,
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
            self.non_causal_encoder = TransformerEncoder(encoder_layer, num_layers=self.cfg.get("non_causal_encoder_num_layers"))
        
        elif self.non_causal_encoder_type.lower() == 'bert':
            # Using bert-large-uncased architecture. 
            # Note: inputs_embeds is supported by BertModel.forward
            # self.non_causal_encoder = BertModel.from_pretrained("bert-large-uncased")
            # bert_dim = self.non_causal_encoder.config.hidden_size # 1024 for large
            
            # # Extend BERT position embeddings to handle sequences > 512
            # # We extend to a safe margin (e.g. 4096) via interpolation
            # current_max_pos = self.non_causal_encoder.config.max_position_embeddings
            # new_max_pos = 4096
            # if current_max_pos < new_max_pos:
            #     logging.info(f"Extending BERT position embeddings from {current_max_pos} to {new_max_pos}")
            #     old_pos_emb = self.non_causal_encoder.embeddings.position_embeddings
            #     new_pos_emb = nn.Embedding(new_max_pos, bert_dim)
                
            #     # Interpolate existing weights to new length
            #     with torch.no_grad():
            #         old_weights = old_pos_emb.weight.data.unsqueeze(0).transpose(1, 2) # (1, H, L)
            #         new_weights = torch.nn.functional.interpolate(old_weights, size=new_max_pos, mode='linear', align_corners=False)
            #         new_pos_emb.weight.data.copy_(new_weights.transpose(1, 2).squeeze(0))
                
            #     # Replace in model
            #     self.non_causal_encoder.embeddings.position_embeddings = new_pos_emb
            #     self.non_causal_encoder.config.max_position_embeddings = new_max_pos
                
            #     # Critical: Update BERT's internal buffers that depend on max_len
            #     # 1. token_type_ids: Should be (1, max_len) for broadcasting
            #     if hasattr(self.non_causal_encoder.embeddings, "token_type_ids"):
            #         new_token_type_ids = torch.zeros((1, new_max_pos), dtype=torch.long, device=self.non_causal_encoder.device)
            #         self.non_causal_encoder.embeddings.register_buffer("token_type_ids", new_token_type_ids, persistent=False)
                
            #     # 2. position_ids: Should be (1, max_len)
            #     if hasattr(self.non_causal_encoder.embeddings, "position_ids"):
            #         new_position_ids = torch.arange(new_max_pos, dtype=torch.long, device=self.non_causal_encoder.device).unsqueeze(0)
            #         self.non_causal_encoder.embeddings.register_buffer("position_ids", new_position_ids, persistent=False)
            from transformers import BertModel
            bert_encoder = BertModel.from_pretrained("bert-large-uncased")
            bert_dim = bert_encoder.config.hidden_size # 1024 for large
            self.non_causal_encoder = bert_encoder.encoder

            if llm_hidden_size != bert_dim:
                self.non_causal_proj_in = nn.Linear(llm_hidden_size, bert_dim)
                self.non_causal_proj_out = nn.Linear(bert_dim, llm_hidden_size)
            del bert_encoder

        elif self.non_causal_encoder_type.lower() == 'whisper':
            from transformers import WhisperModel
            # Using whisper-large-v3 encoder architecture
            # Use Wrapper to support inputs_embeds and FSDP
            whisper = WhisperModel.from_pretrained("openai/whisper-large-v3")
            # self.non_causal_encoder = WhisperEncoderWrapper(
            #     whisper.encoder, 
            #     gradient_checkpointing=self.cfg.get("gradient_checkpointing", False)
            # )
            self.non_causal_encoder = whisper.encoder
            whisper_dim = whisper.config.d_model # 1280
            del whisper
            
            if llm_hidden_size != whisper_dim:
                self.non_causal_proj_in = nn.Linear(llm_hidden_size, whisper_dim)
                self.non_causal_proj_out = nn.Linear(whisper_dim, llm_hidden_size)
        
        else:
            raise ValueError(f"Unknown non_causal_encoder_type: {self.non_causal_encoder_type}")

        # Output projection to match LLM hidden size
        # self.z_projection = torch.nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        
        # New independent head for projecting Z to vocab logits (for alignment)
        # We use embed_tokens.weight.shape[0] to match the actual embedding size (avoiding padding mismatch)
        self.z_head = torch.nn.Linear(
            self.llm.config.hidden_size, 
            self.embed_tokens.weight.shape[0], # Explicitly use Embedding's vocab size
            bias=False
        )

        self._use_fsdp = False
        self._use_tp = False

         # Explicitly set non_causal_encoder to train mode initially
        self.non_causal_encoder.train()
        if self.non_causal_proj_in is not None:
            self.non_causal_proj_in.train()
        if self.non_causal_proj_out is not None:
            self.non_causal_proj_out.train()
        self.z_head.train()

        self.if_force_cot_label = self.cfg.get("if_force_cot_label", False)
        self.if_force_Z_embed = self.cfg.get("if_force_Z_embed", False)
        

    def init_from_model_from_ckpt(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            elif os.path.isdir(checkpoint_path):
                # Handle HuggingFace format directory
                logging.info(f"Loading from HuggingFace format directory: {checkpoint_path}")
                pretrained_model = self.__class__.from_pretrained(checkpoint_path)
                checkpoint_state = pretrained_model.state_dict()
                del pretrained_model
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # partial initialization support
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

    def init_from_fsdp_checkpoint_directory(self, checkpoint_path):
        """
        专门用于加载 FSDP 保存的文件夹格式 Checkpoint (Distributed Checkpoint)。
        只加载模型权重 (state_dict)，忽略优化器状态。
        """
        if checkpoint_path is None:
            return

        # 简单的路径检查
        if not os.path.isdir(checkpoint_path):
            logging.warning(f"FSDP path {checkpoint_path} is not a directory! Skipping load.")
            return

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
             print(f"LOADER: Loading FSDP weights from directory: {checkpoint_path}")

        try:
            # 【核心逻辑】
            # Lightning FSDP 保存时，结构通常是 {"state_dict": model_weights, "optimizer_states": ...}
            # 我们构造一个 mapping，告诉 DCP 把文件里 "state_dict" 下的内容，加载到 self (即当前模型) 中。
            # 这样就完美过滤掉了 optimizer_states，避免了 key missing 的报错。
            
            state_dict_to_load = {"state_dict": self}
            
            dcp.load(
                state_dict=state_dict_to_load,
                checkpoint_id=checkpoint_path,
            )
            
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print("LOADER: Successfully loaded FSDP weights (Optimizer states ignored).")
                
        except Exception as e:
            logging.warning(f"LOADER: Failed to load with 'state_dict' mapping: {e}")
            logging.info("LOADER: Attempting direct model load (in case checkpoint is not wrapped in state_dict)...")
            
            # 备用方案：如果 checkpoint 根目录直接就是权重
            try:
                dcp.load(
                    state_dict=self,
                    checkpoint_id=checkpoint_path,
                )
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print("LOADER: Successfully loaded FSDP weights directly.")
            except Exception as e2:
                 logging.error(f"LOADER: Critical Error loading FSDP checkpoint: {e2}")
                 raise e2


    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS, EOS, and DELAY tokens."""
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        return self._codebook_size + 2

    def forward(
            self,
            input_embeds: Tensor = None,
            cache=None,
            text_labels=None,  # Text token labels for sequential prediction
            semantic_labels=None,  # Semantic token labels
            loss_mask_semantic=None,
            audio_embed=None,    # For training: ELBO audio embeddings
            text_embed=None,     # For training: ELBO text embeddings
            full_loss_mask=None
    ) -> dict[str, Tensor]:
        """
        Sequential text and semantic prediction:
            - Step 1: Text prediction via LLM + lm_head
            - Step 2: Semantic prediction via semantic_predictor using [llm_hidden, text_embed]
        
        Shape annotations:
            input_embeds: (B, T, D) - Input embeddings to LLM
            text_labels: (B, T) - Ground truth text tokens (training only)
            semantic_labels: (B, T) - Ground truth semantic tokens (training only)
            loss_mask: (B, T) - Valid positions mask
        """
        # ========== Step 1: LLM Forward Pass ==========
        # Handle different cache parameter names for different models
        if input_embeds is not None:
            if 'Nemotron' in self.cfg.pretrained_llm:
                # Nemotron uses cache_params instead of past_key_values
                kwargs = {
                    "inputs_embeds": input_embeds,
                    "return_dict": True,
                    "use_cache": cache is not None,
                }
                if cache is not None:
                    kwargs['use_cache'] = True
                    kwargs[self.cfg.get("cache_key", "past_key_values")] = cache
                out = self.llm(**kwargs)
            else:
                out = self.llm(
                    inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
                )

            B, T = input_embeds.shape[:2]
            llm_hidden = out['last_hidden_state']  # (B, T, D)
            text_logits = self.lm_head(llm_hidden)  # (B, T, text_vocab_size)

            ans = {
                "text_logits": text_logits,
                "llm_hidden": llm_hidden,  # Save for inference
            }

            if cache is not None:
                if 'Nemotron' in self.cfg.pretrained_llm:
                    # For Nemotron, get cache from the configured cache key
                    cache_key = self.cfg.get("cache_key", "cache_params")
                    ans["cache"] = getattr(out, cache_key, out.get(cache_key))
                else:
                    # Standard cache handling
                    ans["cache"] = out["past_key_values"]

            # ========== Step 2: Semantic Token Prediction ==========
            # Prepare text embeddings
            # Training: use ground truth text tokens (teacher forcing)
            # Inference: handled separately in generate method
            if text_labels is not None:
                # text_labels: (B, T)
                text_embeds = self.embed_tokens(text_labels)  # (B, T, D)
            else:
                # Fallback: use predicted text tokens
                # text_logits: (B, T, text_vocab_size) -> argmax -> (B, T)
                predicted_text_tokens = text_logits.argmax(dim=-1)  # (B, T)
                text_embeds = self.embed_tokens(predicted_text_tokens)  # (B, T, D)

            # Call semantic predictor
            # Inputs:
            #   llm_hidden: (B, T, D)
            #   text_embeds: (B, T, D)
            # Outputs:
            #   semantic_logits: (B, T, semantic_vocab_size=16384)
            semantic_result = self.semantic_predictor(
                llm_hidden=llm_hidden,
                text_embeds=text_embeds,
                semantic_labels=semantic_labels,  # (B, T) or None
                loss_mask=loss_mask_semantic,  # (B, T) or None
            )
            
            ans["semantic_logits"] = semantic_result["semantic_logits"]  # (B, T, 16384)
            if "semantic_loss" in semantic_result:
                ans["semantic_loss"] = semantic_result["semantic_loss"]  # scalar
            ans["cot_classification_logits"] = self.cot_classification_head(out['last_hidden_state'])
        else:
             # Training mode: use ELBO logic with audio_embed + text_embed
            B, T = audio_embed.shape[:2]  # T is full T frames
            T = T - 1  # LLM processes T-1 frames

            # Step 1: Generate Z using non-causal TransformerEncoder (full T frames)
            encoder_input = audio_embed + text_embed
            
            # Apply input projection if needed
            if self.non_causal_proj_in is not None:
                encoder_input = self.non_causal_proj_in(encoder_input)

            if 'trans' in self.non_causal_encoder_type.lower():
                Z = self.non_causal_encoder(encoder_input)  # (B, T_full, H)
            elif self.non_causal_encoder_type == 'bert':
                # BERT forward with inputs_embeds
                outputs = self.non_causal_encoder(encoder_input)
                Z = outputs.last_hidden_state
            
            # Apply output projection if needed
            if self.non_causal_proj_out is not None:
                Z = self.non_causal_proj_out(Z)

            # Z = self.z_projection(Z)  # (B, T_full, H)

            # --- Modified Logic: Use Soft Embedding from Z ---
            # Project Z to vocab logits (using independent z_head to match embedding dimension)
            z_logits = self.z_head(Z) # (B, T_full, V_embed)
            z_probs = torch.softmax(z_logits, dim=-1) # (B, T_full, V_embed)
            
            # Compute Soft Embedding: E_z = Probs @ EmbeddingMatrix
            # Use self.embed_tokens directly as it was moved out of self.llm in __init__
            embed_weight = self.embed_tokens.weight
            
            E_z = z_probs @ embed_weight # (B, T_full, H)
            # ------------------------------------------------

            # Step 2: Construct LLM input (take first T-1 frames) - optimized for memory
            mask_input = full_loss_mask[:, :-1].unsqueeze(-1)  # (B, T-1, 1)

            # Use in-place operations to avoid creating intermediate tensors
            llm_input = audio_embed[:, :-1].clone() * self.cfg.get("duplex_user_channel_weight", 1.0) # Start with audio base

            # Add masked text contribution safely (don't modify original tensors)
            llm_input.addcmul_(text_embed[:, :-1], mask_input)  # llm_input += text * mask

            # Add masked Z contribution in-place
            # Use Soft Embedding E_z instead of raw Z
            llm_input.addcmul_(E_z[:, :-1], 1 - mask_input)  # llm_input += E_z * (1-mask)

            # Clean up intermediate tensors
            del mask_input

            # Step 3: Forward through LLM
            out = self.llm(
                inputs_embeds=llm_input,
                past_key_values=cache,
                use_cache=cache is not None,
                output_hidden_states=True,
                return_dict=True
            )

            B, T = llm_input.shape[:2]
            llm_hidden = out['last_hidden_state']  # (B, T, D)
            text_logits = self.lm_head(llm_hidden)  # (B, T, text_vocab_size)

            ans = {
                "text_logits": text_logits,
                "llm_hidden": llm_hidden,  # Save for inference
            }

            if cache is not None:
                if 'Nemotron' in self.cfg.pretrained_llm:
                    # For Nemotron, get cache from the configured cache key
                    cache_key = self.cfg.get("cache_key", "cache_params")
                    ans["cache"] = getattr(out, cache_key, out.get(cache_key))
                else:
                    # Standard cache handling
                    ans["cache"] = out["past_key_values"]

            # ========== Step 2: Semantic Token Prediction ==========
            # Prepare text embeddings
            # Training: use ground truth text tokens (teacher forcing)
            # Inference: handled separately in generate method
            if text_labels is not None:
                # text_labels: (B, T)
                text_embeds = self.embed_tokens(text_labels)  # (B, T, D)
            else:
                # Fallback: use predicted text tokens
                # text_logits: (B, T, text_vocab_size) -> argmax -> (B, T)
                predicted_text_tokens = text_logits.argmax(dim=-1)  # (B, T)
                text_embeds = self.embed_tokens(predicted_text_tokens)  # (B, T, D)

            # Call semantic predictor
            # Inputs:
            #   llm_hidden: (B, T, D)
            #   text_embeds: (B, T, D)
            # Outputs:
            #   semantic_logits: (B, T, semantic_vocab_size=16384)
            semantic_result = self.semantic_predictor(
                llm_hidden=llm_hidden,
                text_embeds=text_embeds,
                semantic_labels=semantic_labels,  # (B, T) or None
                loss_mask=loss_mask_semantic,  # (B, T) or None
            )
            
            ans["semantic_logits"] = semantic_result["semantic_logits"]  # (B, T, 16384)
            if "semantic_loss" in semantic_result:
                ans["semantic_loss"] = semantic_result["semantic_loss"]  # scalar

            # Compute MSE loss inside forward to avoid keeping Z in memory
            mse_loss = 0.0
            if full_loss_mask is not None:
                
                min_seq_len = min(llm_hidden.shape[1], Z.shape[1] - 1)

                # Apply MSE loss only where loss_mask is 0 (non-response regions)
                loss_mask_truncated = full_loss_mask[:, 1:min_seq_len+1]  # (B, min_T)
                mse_weight = (1 - loss_mask_truncated).flatten(0, 1)  # (B*min_T)

                # Compute MSE loss
                # mse_loss_raw = torch.nn.functional.mse_loss(
                #     Z[:, 1:min_seq_len+1, :].flatten(0, 1).detach(), last_layer_hidden[:, :min_seq_len, :].flatten(0, 1), reduction='none'
                # ).mean(-1)  # (B*min_T)
                # mse_loss = (mse_loss_raw * mse_weight).sum()

                # Prepare flattened tensors
                
                # Use KL Divergence Loss instead of MSE/Cosine
                # P_z (target) comes from z_logits (Dim: 151667)
                # P_1 (prediction) comes from text_logits (Dim: 152064)

                target_logits = z_logits[:, 1:min_seq_len+1, :].flatten(0, 1).detach() # (N, V_embed)
                pred_logits = text_logits[:, :min_seq_len, :].flatten(0, 1) # (N, V_padded)

                # TRUNCATE pred_logits or PAD target_logits?
                # Option B (Better): Pad target_logits to match pred_logits dimension
                # This ensures we don't ignore probability mass that LLM might assign to padding tokens
                vocab_size_target = target_logits.size(-1)
                vocab_size_pred = pred_logits.size(-1)
                
                if vocab_size_pred > vocab_size_target:
                    pad_len = vocab_size_pred - vocab_size_target
                    # Pad with -inf so that softmax probability is 0
                    target_logits = torch.nn.functional.pad(
                        target_logits, (0, pad_len), value=float('-inf')
                    )
                elif vocab_size_pred < vocab_size_target:
                     # This should theoretically not happen given Qwen structure, but for safety:
                     pred_logits = torch.nn.functional.pad(
                        pred_logits, (0, vocab_size_target - vocab_size_pred), value=float('-inf')
                    )

                # Compute KL(P_z || P_1)
                # target is probabilities (softmax of target_logits)
                # input is log_probabilities (log_softmax of pred_logits)
                
                target_probs = torch.softmax(target_logits, dim=-1)
                pred_log_probs = torch.log_softmax(pred_logits, dim=-1)
                
                kl_loss = torch.nn.functional.kl_div(pred_log_probs, target_probs, reduction='none').sum(-1) # (N)
                
                mse_loss = (kl_loss * mse_weight).sum()
                

            ans["cot_classification_logits"] = self.cot_classification_head(out['last_hidden_state'])
            ans["mse_loss"] = mse_loss  # MSE loss computed internally using last_hidden_state

        return ans


    def prepare_inputs(self, batch: dict):
        """
        Prepare training inputs, handling alignment of text and semantic tokens.
        
        Key design:
        1. Text autoregressive: input[t-1] -> predict text[t]
        2. Semantic sequential prediction: use text[t] -> predict semantic[t]
        3. Therefore semantic_labels and text_labels are aligned (same timestep)
        
        Shape flow:
        - source_encoded: (B, T_audio, D) - Audio perception features
        - target_tokens: (B, T_text) - Text tokens
        - target_semantic: (B, T_semantic) - Semantic tokens (extracted from target_audio)
        - Align to same length min_len
        - Shift operation:
            text_inputs: target_tokens[:, :-1]  -> (B, T-1)
            text_labels: target_tokens[:, 1:]   -> (B, T-1)
            semantic_labels: target_semantic[:, 1:]  -> (B, T-1)  # Note: aligned with text_labels!
        """
        source_encoded, source_encoded_lens, asr_emb = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
            return_encoder_emb=True,
        )

        target_tokens = batch["target_tokens"]  # (B, T_text)
        loss_mask = batch.get("loss_mask", None)
        
        # ========== Extract Target Semantic Tokens (agent speech) ==========

        target_semantic = extract_speech_token(
            self.whispervq,
            self.feature_extractor,
            [(batch["target_audio"][i].unsqueeze(0), self.target_sample_rate) for i in range(batch["target_audio"].shape[0])],
        )
        target_semantic = torch.tensor(target_semantic, dtype=torch.long, device=self.device)  # (B, T_tgt_audio)

        # ========== Length Alignment ==========
      
        min_len = min(
            source_encoded.shape[1],      # T_perception
            target_semantic.shape[1],     # T_tgt_audio
            target_tokens.shape[1],        # T_text
            loss_mask.shape[1]        # T_text
        )
        
       
        source_encoded = source_encoded[:, :min_len]          # (B, min_len, D)
        target_semantic = target_semantic[:, :min_len]        # (B, min_len)
        target_tokens = target_tokens[:, :min_len]            # (B, min_len)
        loss_mask = loss_mask[:, :min_len]                    # (B, min_len)
        source_encoded_lens = torch.clamp_(source_encoded_lens, max=min_len)  # (B,)

        # ========== Apply Autoregressive Shift ==========
        # Text channel: input[t-1] -> predict text[t]
        text_inputs = target_tokens[:, :-1]   # (B, T-1) - Used as LLM input
        text_labels = target_tokens[:, 1:]    # (B, T-1) - text prediction target
        if loss_mask is not None:
            full_loss_mask = loss_mask.clone()  # Keep full T frames for ELBO
            loss_mask = loss_mask[:, 1:]
        else:
            full_loss_mask = None
        
        # Semantic channel: text[t] -> predict semantic[t]
        # Key: semantic_labels and text_labels are aligned (predicting same timestep)
        # In forward, we use text_labels as source for text_embeds
        semantic_labels = target_semantic[:, 1:]  # (B, T-1) - semantic prediction target

        # ========== Prepare Input Embeddings ==========
        # Combine text_embeds and audio perception embeds
        text_embeds = self.embed_tokens(text_inputs)  # (B, T-1, D)
        audio_embeds = source_encoded[:, :-1]         # (B, T-1, D)
        
        # ========== Ensure dtype consistency ==========
        if text_embeds.dtype != audio_embeds.dtype:
            audio_embeds = audio_embeds.to(dtype=text_embeds.dtype)
        
        # ========== Apply Gated Fusion or Fixed Weight ==========
        # if self.gated_fusion is not None:  #   不支持！！！！！！！！！！！！
        #     # Create PAD mask for token-aware gating
        #     is_pad_mask = (text_inputs == self.text_pad_id)  # (B, T-1)
        #     input_embeds = self.gated_fusion(text_embeds, audio_embeds, is_pad_mask)
        # else:
        #     # Original behavior: fixed weight addition
        #     input_embeds = text_embeds + audio_embeds * self.cfg.get("duplex_user_channel_weight", 1.0)
        
        # ========== Prepare Loss Mask ==========
        # loss_mask: (B, T-1) - True indicates valid positions, False indicates padding
        loss_mask_semantic = torch.ones_like(text_labels, device=self.device, dtype=torch.bool)  # (B, T-1)
        text_embed_input = self.embed_tokens(target_tokens)  # (B, T-1, D)
        audio_embed_input = source_encoded        # (B, T-1, D)
        result = {
            # "input_embeds": input_embeds,           # (B, T-1, D)
            "input_lens": source_encoded_lens - 1,  # (B,) - Subtract 1 due to shift operation
            "output_lens": source_encoded_lens - 1,  # (B,)
            "text_labels": text_labels,             # (B, T-1)
            "semantic_labels": semantic_labels,     # (B, T-1)
            "loss_mask_semantic": loss_mask_semantic, # (B, T-1)
            "loss_mask": loss_mask,                 # (B, T-1)
            "cot_label": loss_mask.int(),
            "full_loss_mask": full_loss_mask,       # (B, T)
            "text_embed": text_embed_input,              # (B, T-1, D)
            "audio_embed": audio_embed_input,            # (B, T-1, D)
        }

        return result

    def training_step(self, batch: dict, batch_idx: int):
        """
        Training step for duplex S2S model.
        
        Data flow:
        1. prepare_inputs: batch -> input_embeds, text_labels, semantic_labels, loss_mask
        2. forward: input_embeds -> text_logits, semantic_logits
        3. Loss computation: text_loss + semantic_loss
        """
        # Set frozen modules to eval mode
        # Note: semantic_predictor is NOT included as it should be trainable
        frozen_modules = [self.perception.preprocessor, self.perception.encoder, self.llm]
        
        for m in frozen_modules:
            if is_frozen(m):
                m.eval()

        res = {"learning_rate": torch.as_tensor(
            self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)}

        if batch["audio_data"] is not None:
            inputs = self.prepare_inputs(batch["audio_data"])

            # ========== Forward Pass ==========
            forward_outputs = self(
                # inputs["input_embeds"],              # (B, T-1, D)
                text_labels=inputs["text_labels"],   # (B, T-1)
                semantic_labels=inputs["semantic_labels"],  # (B, T-1)
                loss_mask_semantic=inputs["loss_mask_semantic"],
                text_embed=inputs["text_embed"],
                audio_embed=inputs["audio_embed"],
                full_loss_mask=inputs["full_loss_mask"],
            )

            num_frames = inputs["input_lens"].sum()

            with loss_parallel():
                text_logits = forward_outputs["text_logits"]  # (B, T-1, text_vocab_size)

                loss_mask = inputs['loss_mask']
                # if loss_mask is None:
                #     loss_mask = torch.ones_like(inputs["text_labels"]).to(inputs["text_labels"].device)

                # ========== Calculate Text Loss ==========
                text_loss = (
                    (torch.nn.functional.cross_entropy(
                        text_logits.flatten(0, 1),  # (B*(T-1), V_text)
                        inputs["text_labels"].flatten(0, 1),  # (B*(T-1),)
                        reduction="none",
                    ) * loss_mask.flatten(0, 1)).sum(-1) / num_frames
                )

                # ========== Calculate Semantic Loss & Accuracy ==========
                semantic_loss = forward_outputs.get("semantic_loss", torch.tensor(0.0, device=text_logits.device))
                
                # Calculate semantic accuracy
                semantic_logits = forward_outputs["semantic_logits"]  # (B, T-1, 16384)
                semantic_labels = inputs["semantic_labels"]  # (B, T-1)
                loss_mask_semantic = inputs["loss_mask_semantic"]  # (B, T-1)
                
                if loss_mask_semantic is not None:
                    semantic_pred = semantic_logits[loss_mask_semantic].argmax(-1)  # (N,)
                    semantic_target = semantic_labels[loss_mask_semantic]  # (N,)
                    semantic_acc = (semantic_pred == semantic_target).float().mean()
                else:
                    semantic_pred = semantic_logits.argmax(dim=-1)  # (B, T-1)
                    semantic_acc = (semantic_pred == semantic_labels).float().mean()

                # ========== Calculate Text Accuracy ==========
                with torch.no_grad():
                    predicted_tokens = torch.argmax(text_logits * loss_mask.unsqueeze(-1) , dim=-1)# (B, T-1)
                    target_tokens = inputs["text_labels"] * loss_mask.unsqueeze(-1)  # (B, T-1)
                    valid_mask = (target_tokens != self.text_pad_id) * loss_mask.unsqueeze(-1)

                    correct_predictions = (predicted_tokens == target_tokens) & valid_mask

                    if valid_mask.sum() > 0:
                        token_accuracy = correct_predictions.sum().float() / valid_mask.sum().float()
                    else:
                        token_accuracy = torch.tensor(0.0, device=text_logits.device)

                # ========== Combined Loss ==========
                loss = self.cfg.text_loss_weight * text_loss + self.cfg.get("audio_loss_weight", 0) * semantic_loss

                mse_loss = forward_outputs.get("mse_loss", 0.0) / num_frames  # Normalize by num_frames

                # Add MSE reconstruction loss to total loss
                mse_loss_weight = self.cfg.get("mse_reconstruction_loss_weight", 1.0)
                loss = loss + mse_loss_weight * mse_loss

                # Add binary classification loss for COT/response classification
                cot_classification_loss = 0.0
                if 'cot_classification_logits' in forward_outputs and 'cot_label' in inputs:
                    cot_logits = forward_outputs['cot_classification_logits']  # (B, T, 1)
                    cot_labels = inputs['cot_label'].float()  # (B, T)

                    # Ensure shapes match
                    min_seq_len = min(cot_logits.shape[1], cot_labels.shape[1])
                    cot_logits_truncated = cot_logits[:, :min_seq_len, :]  # (B, min_T, 1)
                    cot_labels_truncated = cot_labels[:, :min_seq_len]     # (B, min_T)

                    # Flatten for loss computation
                    cot_logits_flat = cot_logits_truncated.squeeze(-1)  # (B, min_T)
                    cot_labels_flat = cot_labels_truncated              # (B, min_T)

                    cot_classification_loss = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            cot_logits_flat.flatten(0, 1),  # (B, T) -> (*)
                            cot_labels_flat.flatten(0, 1),
                            reduction="none",
                        )
                    ).sum(-1) / (num_frames)
                # Add COT classification loss to total loss
                cot_loss_weight = self.cfg.get("cot_classification_loss_weight", 1.0)
                loss = loss + cot_loss_weight * cot_classification_loss



                B, T = inputs["text_labels"].shape[:2]
                ans = {
                    "audio_loss": loss,
                    "audio_to_text_loss": text_loss,
                    "audio_to_semantic_loss": semantic_loss,
                    "mse_loss": mse_loss,
                    "cot_classification_loss": cot_classification_loss,
                    "batch": B,
                    "length": T,
                    "token_accuracy": token_accuracy,
                    "semantic_token_accuracy": semantic_acc,
                }

                res.update(ans)

        if batch["text_data"] is not None:
            text_input_ids = batch["text_data"]["text_tokens"][:, :-1]
            text_target = batch["text_data"]["text_tokens"][:, 1:]

            text_out = self.llm(
                inputs_embeds=self.embed_tokens(text_input_ids),
                past_key_values=None,
                use_cache=False,
                return_dict=True,
            )
            text_logits = self.lm_head(text_out['last_hidden_state'])  # (B, T, Vt)

            text_loss = torch.nn.functional.cross_entropy(
                text_logits.flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                text_target.flatten(0, 1),
                ignore_index=self.text_pad_id,
            )
            res.update(
                {
                    "text_to_text_loss": text_loss,
                }
            )

        res["loss"] = (1. - self.cfg.text_to_text_loss_weight) * res.get("audio_loss", 0.0) + \
                      self.cfg.text_to_text_loss_weight * res.get("text_to_text_loss", 0.0)
        self.log_dict(res, on_step=True)

        return res

    def on_train_epoch_start(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        # Initialize ResultsLogger (it will automatically find manifest_files in its own directory)
        self.results_logger = ResultsLogger(self.validation_save_path).reset()

        self.bleu = BLEU().reset()

        # Initialize turn taking metrics
        self.turn_taking_metrics = TurnTakingMetrics(
            eos_token_id=self.text_eos_id,
            bos_token_id=self.text_bos_id,
            tolerance=13,
            latency_multiplier=0.08
        ).reset()

    def on_validation_epoch_end(self, prefix="val") -> None:

        bleu = self.bleu.compute()
        for k, m in bleu.items():
            if "qa" not in k and "mmsu" not in k:
                self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        acc_metrics = self.results_logger.compute_and_save()

        for name, result_dict in acc_metrics.items():
            # Log regular accuracy for QA datasets
            if 'acc' in result_dict:
                self.log(f"{prefix}_{name}_acc", result_dict['acc'].to(self.device), on_epoch=True, sync_dist=True)

            # Log MCQ accuracy for MCQ datasets
            if 'mcq_acc' in result_dict:
                self.log(f"{prefix}_{name}_mcq_acc", result_dict['mcq_acc'].to(self.device), on_epoch=True,
                         sync_dist=True)

        # Log turn taking metrics
        turn_taking_metrics = self.turn_taking_metrics.compute()
        for k, m in turn_taking_metrics.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Validation step: 
        - Only use flow matching to generate audio samples for results_logger in first few batches
        - Does not compute mos and asr_bleu
        """

        decode_audio = (batch_idx < 2) and self.cfg.get("pretrained_flow", None) is not None

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            dataset_batch = dataset_batch["audio_data"]
            
            get_input_dict = self.prepare_inputs(dataset_batch)
            loss_mask = get_input_dict['loss_mask']
            if self.if_force_Z_embed:
                text_embed = get_input_dict['text_embed']
            else:
                text_embed = None
            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
                decode_audio=decode_audio,
                loss_mask = loss_mask,
                text_embed = text_embed
            )
            print('if_force_cot_label', self.if_force_cot_label)
            print('if_force_Z_embed', self.if_force_Z_embed)

      
            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])

            # Update turn taking metrics
            if "source_tokens" in dataset_batch and results["tokens_text"] is not None:
                self.turn_taking_metrics.update(
                    name=name,
                    source_tokens=dataset_batch["source_tokens"],
                    pred_tokens=results["tokens_text"]
                )


            if "audio" in results:

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=None,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
                    pred_audio_sr=22050,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                )
            else:

                fake_pred_audio, fake_audio_len = self._generate_fake_audio_from_tokens(results["tokens_text"])

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=None,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=fake_pred_audio,
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                )

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def on_predict_epoch_start(self) -> None:
        return self.on_train_epoch_start()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        batch = batch["audio_data"]

        force_bos_positions = None
        force_bos_num_tokens_after_user_eos = self.cfg.prediction.get("force_bos_num_tokens_after_user_eos", None)
        if force_bos_num_tokens_after_user_eos is not None:
            force_bos_positions = []
            for cur_source_tokens in batch["source_tokens"]:
                tmp = torch.where(cur_source_tokens == self.text_eos_id)[0]
                if len(tmp) > 0:
                    force_bos_positions.append(tmp[0].item() + force_bos_num_tokens_after_user_eos)
                else:
                    force_bos_positions.append(None)
        loss_mask = self.prepare_inputs(batch)['loss_mask']
        prediction = self.offline_inference(
            batch["source_audio"],
            batch["source_audio_lens"],
            decode_audio=self.cfg.prediction.decode_audio,
            input_pad_len=self.cfg.prediction.max_new_seconds * self.cfg.prediction.input_sample_rate,
            force_bos_positions=force_bos_positions,
            loss_mask = loss_mask,
        )
        print('if_force_cot_label', self.if_force_cot_label)
        prediction["sample_id"] = batch["sample_id"]
        return prediction

    def _cal_acc(self, pad_outputs, pad_targets, ignore_label):
        """Calculate accuracy for predictions, ignoring specified label."""
        pad_pred = pad_outputs.argmax(-1)
        mask = pad_targets != ignore_label
        numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
        denominator = torch.sum(mask)
        return (numerator / denominator).detach().item() if denominator > 0 else 0.0

    def load_flow_decoder(self):
        """Load flow-based audio decoder for waveform generation."""
        if hasattr(self, 'audio_decoder'):
            return  # Already loaded
            
        from nemo.collections.speechlm2.modules.flow_inference import AudioDecoder
        
        flow_config = os.path.join(self.cfg.pretrained_flow, "config.yaml")
        flow_checkpoint = os.path.join(self.cfg.pretrained_flow, 'flow.pt')
        hift_checkpoint = os.path.join(self.cfg.pretrained_flow, 'hift.pt')
        
        self.audio_decoder = AudioDecoder(
            config_path=flow_config,
            flow_ckpt_path=flow_checkpoint,
            hift_ckpt_path=hift_checkpoint
        )

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    def _generate_fake_audio_from_tokens(self, tokens_text: torch.Tensor):
        """
        Generate fake audio based on text tokens for visualization.

        Logic:
        - Default value: 0
        - After first text_bos_id: 1
        - After first text_eos_id: back to 0
        - text_pad_id between bos and eos: 0.5
        - text_pad_id elsewhere: 0

        Args:
            tokens_text: (batch_size, seq_len) tensor

        Returns:
            fake_audio: (batch_size, audio_len) tensor
            audio_lengths: (batch_size,) tensor with audio lengths
        """
        batch_size, seq_len = tokens_text.shape
        token_duration = 0.08  # seconds per token
        samples_per_token = int(token_duration * self.target_sample_rate)
        audio_len = seq_len * samples_per_token

        # Initialize fake audio tensor
        fake_audio = torch.zeros(batch_size, audio_len, device=tokens_text.device, dtype=torch.float32)
        audio_lengths = torch.full((batch_size,), audio_len, device=tokens_text.device, dtype=torch.long)

        for b in range(batch_size):
            current_tokens = tokens_text[b].cpu().numpy()  # Convert to numpy for easier processing
            audio_values = torch.zeros(seq_len, device=tokens_text.device, dtype=torch.float32)

            in_speech = False  # Track whether we're between bos and eos

            for t in range(seq_len):
                token_id = int(current_tokens[t])

                if token_id == self.text_bos_id:
                    # Start of speech
                    in_speech = True
                    audio_values[t] = 1.0
                elif token_id == self.text_eos_id:
                    # End of speech
                    in_speech = False
                    audio_values[t] = 0.0
                elif token_id == self.text_pad_id:
                    if in_speech:
                        # Pad token between bos and eos (after text generation)
                        audio_values[t] = 0.5
                    else:
                        # Pad token outside speech
                        audio_values[t] = 0.0
                else:
                    # Regular text token
                    if in_speech:
                        audio_values[t] = 1.0
                    else:
                        audio_values[t] = 0.0

            # Expand each token value to cover the corresponding audio samples
            for t in range(seq_len):
                start_sample = t * samples_per_token
                end_sample = min((t + 1) * samples_per_token, audio_len)
                fake_audio[b, start_sample:end_sample] = audio_values[t]

        return fake_audio, audio_lengths

    @torch.no_grad()
    def offline_inference(
            self,
            input_signal: torch.Tensor,
            input_signal_lens: torch.Tensor,
            decode_audio: bool = True,
            input_pad_len: int = 0,
            force_bos_positions=None,
            loss_mask = None,
            text_embed = None
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive text prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: unused, kept for interface compatibility.
            input_pad_len: padding length for input signal.
            force_bos_positions: optional positions to force BOS token generation.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "source_audio": input audio signal.
                * "source_audio_len": input audio lengths.
        """

        if self.cfg.get("custom_sample_inference", None):
            device = input_signal.device
            input_signal, sr = torchaudio.load(self.cfg.custom_sample_inference)
            input_signal = input_signal.to(device)[:1, :]
            input_signal = resample(input_signal, sr, self.source_sample_rate)
            input_signal_lens = torch.tensor([input_signal.size(-1)]).to(device)

        if force_bos_positions is not None:
            assert input_signal.shape[0] == len(
                force_bos_positions), "force_bos_positions must have the same length as batch size"

        if input_pad_len > 0:
            input_signal = torch.nn.functional.pad(input_signal, (0, input_pad_len), mode='constant', value=0)
            input_signal_lens = input_signal_lens + input_pad_len

        source_encoded, lengths, asr_emb = self.perception(
            input_signal=input_signal, input_signal_length=input_signal_lens, return_encoder_emb=True
        )
        
        E_z = None
        if text_embed is not None and self.if_force_Z_embed and self.if_force_cot_label:
            min_len = min(source_encoded.shape[1], text_embed.shape[1])
            source_encoded = source_encoded[:, :min_len, :]
            text_embed = text_embed[:, :min_len, :]
            # Step 1: Generate Z using non-causal TransformerEncoder (full T frames)
            encoder_input = source_encoded + text_embed
            
            # Apply input projection if needed
            if self.non_causal_proj_in is not None:
                encoder_input = self.non_causal_proj_in(encoder_input)

            if 'trans' in self.non_causal_encoder_type.lower():
                Z = self.non_causal_encoder(encoder_input)  # (B, T_full, H)
            elif self.non_causal_encoder_type == 'bert':
                # BERT forward with inputs_embeds
                outputs = self.non_causal_encoder(encoder_input)
                Z = outputs.last_hidden_state
            
            # Apply output projection if needed
            if self.non_causal_proj_out is not None:
                Z = self.non_causal_proj_out(Z)

            # Z = self.z_projection(Z)  # (B, T_full, H)

            # --- Modified Logic: Use Soft Embedding from Z ---
            # Project Z to vocab logits (using independent z_head to match embedding dimension)
            z_logits = self.z_head(Z) # (B, T_full, V_embed)
            z_probs = torch.softmax(z_logits, dim=-1) # (B, T_full, V_embed)
            
            # Compute Soft Embedding: E_z = Probs @ EmbeddingMatrix
            # Use self.embed_tokens directly as it was moved out of self.llm in __init__
            embed_weight = self.embed_tokens.weight
            
            E_z = z_probs @ embed_weight # (B, T_full, H)
        # if self.global_rank == 0:
        #     print('loss_mask', loss_mask.shape)
        #     print('source_encoded', source_encoded.shape)
        #     print('T_local', T_local)
        #     print(self._use_fsdp)
        # Determine decoding length and pad if FSDP
        B, T_local, H = source_encoded.shape
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=source_encoded.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                last_frame_source = source_encoded[:, T_local - 1: T_local, :]
                pad_source = last_frame_source.repeat(1, T - T_local, 1)
                source_encoded = torch.cat([source_encoded, pad_source], dim=1)
                last_frame_asr = asr_emb[:, T_local - 1: T_local, :]
                pad_asr = last_frame_asr.repeat(1, T - T_local, 1)
                asr_emb = torch.cat([asr_emb, pad_asr], dim=1)
            
            if loss_mask is not None and T > loss_mask.shape[1]:
                pad_mask = torch.ones((loss_mask.shape[0], T - loss_mask.shape[1]), device=source_encoded.device).to(torch.long)
                loss_mask = torch.cat([loss_mask, pad_mask], dim = 1)
            if E_z is not None and T > E_z.shape[1]:
                pad_E_z = E_z[:, :T_local, :][:, T_local - 1: T_local, :].repeat(1, T - T_local, 1)
                E_z = torch.cat([E_z, pad_E_z], dim=1)
                # if self.global_rank == 0:
                #     print('pad_mask', pad_mask.shape)
        else:
            T = T_local

        # This cache is for self.llm
        use_cache = True
        if 'Nemotron' in self.cfg.pretrained_llm:
            # For Nemotron, due to cache issues, we disable cache and use full history mode
            cache = None
            use_cache = False
            logging.info("Using no-cache mode for Nemotron (full history each step)")
        else:
            # Standard cache for other models
            cache = DynamicCache()
            use_cache = True

        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_semantic = torch.empty(B, T, device=self.device, dtype=torch.long)
        
        # For TransformerSemanticPredictor: collect llm_hidden history
        # This is needed because in cached mode, ans["llm_hidden"] only contains current step
        if self._use_transformer_semantic:
            llm_hidden_history = torch.empty(B, T, self.llm.config.hidden_size, device=self.device, dtype=source_encoded.dtype)
        
        # ========== Unified Autoregressive Loop (Sequential Prediction) ==========
        # Design:
        # 1. Each timestep first predicts text token via LLM
        # 2. Then use [llm_hidden, text_embed] to predict semantic token
        # 3. This way semantic depends on same-timestep text (sequential prediction)
        gen_cot_classification = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_cot_logits = torch.empty(B, T, 1, device=self.device, dtype=torch.float)
        prev_hidden_state = None
        history_input_emb = []
        
        for t in tqdm(range(T)):
            # ===== Step 1: Prepare Input Embeddings =====
            # Get previous token: PAD for first step, generated token for subsequent steps
            if t == 0:
                last_text_token = torch.full((B,), fill_value=self.text_pad_id, device=self.device, dtype=torch.long)
            else:
                last_text_token = gen_text[:, t - 1]
            
            last_text_emb = self.embed_tokens(last_text_token)  # (B, D)
            
            # Force BOS at specific positions if requested
            if force_bos_positions is not None:
                for batch_idx in range(last_text_emb.shape[0]):
                    if force_bos_positions[batch_idx] == t and not (gen_text[batch_idx, :t] == self.text_bos_id).any():
                        last_text_emb[batch_idx] = self.embed_tokens(
                            torch.full((1,), fill_value=self.text_bos_id, device=self.device))

            current_audio_emb = source_encoded[:, t:t+1]  # (B, 1, D)
            
            # Ensure dtype consistency
            if last_text_emb.dtype != current_audio_emb.dtype:
                current_audio_emb = current_audio_emb.to(dtype=last_text_emb.dtype)
            
            # Apply gated fusion or fixed weight
            # if self.gated_fusion is not None:
            #     is_pad_mask = (last_text_token == self.text_pad_id).unsqueeze(-1)  # (B, 1)
            #     current_input_emb = self.gated_fusion(
            #         last_text_emb.unsqueeze(1),  # (B, 1, D)
            #         current_audio_emb,
            #         is_pad_mask
            #     )
            # else:
            #     current_input_emb = last_text_emb.unsqueeze(1) + current_audio_emb * self.cfg.get("duplex_user_channel_weight", 1.0)

            if t == 0:
                current_input_emb = last_text_emb.unsqueeze(1) + current_audio_emb * self.cfg.get("duplex_user_channel_weight", 1.0)
            else:
                prev_cot_class = gen_cot_classification[:, t - 1]  # (B,)
                # Get previous step's hidden state (from last forward pass)
                # prev_hidden_state = ans["hidden_states"][-1][:, -1, :]  # (B, H) - last layer, last time step
                # hidden_logits = self.llm.lm_head(prev_hidden_state) # (B, V_embed)
                if E_z is None:
                    hidden_logits = ans["text_logits"][:, -1] # (B, V_embed)
                    hidden_probs = torch.softmax(hidden_logits, dim=-1) # (B, V_embed)

                    # Use self.embed_tokens directly
                    embed_weight = self.embed_tokens.weight

                    soft_hidden_emb = hidden_probs[:, :self.embed_tokens.weight.shape[0]]  @ embed_weight # (B, H)
                else:
                    soft_hidden_emb = E_z[:, t, :]
                chosen_emb = torch.where(
                    prev_cot_class.unsqueeze(-1) == 1,  # (B, 1)
                    last_text_emb,                           # (B, H) - text token embedding
                    soft_hidden_emb                     # (B, H) - soft embedding from hidden state
                )
                current_input_emb = chosen_emb.unsqueeze(1) + current_audio_emb * self.cfg.get("duplex_user_channel_weight", 1.0)


            # ===== Step 2: LLM Forward Pass (Text Prediction) =====

            if use_cache:
                # Standard cached mode - pass only current step
                cache_to_use = cache if t == 0 else ans["cache"]
                ans = self(current_input_emb, cache=cache_to_use)  # Only predict text, no semantic_labels
                gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)  # (B,)
            else:
                # No-cache mode for Nemotron - pass full history up to current step
                if t == 0:
                    # First step: no history to reconstruct
                    ans = self(current_input_emb, cache=None)
                    gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
                else:
                    # # Reconstruct full history from step 0 to t-1
                    # if self.gated_fusion is not None:
                    #     all_text_tokens = gen_text[:, :t]  # (B, t)
                    #     all_text_emb = self.embed_tokens(all_text_tokens)  # (B, t, D)
                    #     all_audio_emb = source_encoded[:, :t]  # (B, t, D)
                        
                    #     # Ensure dtype consistency
                    #     if all_text_emb.dtype != all_audio_emb.dtype:
                    #         all_audio_emb = all_audio_emb.to(dtype=all_text_emb.dtype)
                        
                    #     # Create PAD mask for history
                    #     is_pad_mask_history = (all_text_tokens == self.text_pad_id)  # (B, t)
                    #     history_input_emb = self.gated_fusion(all_text_emb, all_audio_emb, is_pad_mask_history)
                    #     full_input_emb = torch.cat([history_input_emb, current_input_emb], dim=1)  # (B, t+1, D)
                    # else:
                    #     # Fixed weight approach
                    #     all_text_tokens = gen_text[:, :t]
                    #     all_text_emb = self.embed_tokens(all_text_tokens)
                    #     all_audio_emb = source_encoded[:, :t]
                        
                    #     # Ensure dtype consistency
                    #     if all_text_emb.dtype != all_audio_emb.dtype:
                    #         all_audio_emb = all_audio_emb.to(dtype=all_text_emb.dtype)    
                        
                    #     history_input_emb = all_text_emb + all_audio_emb * self.cfg.get("duplex_user_channel_weight", 1.0)
                    #     full_input_emb = torch.cat([history_input_emb, current_input_emb], dim=1)

                    full_input_emb = torch.cat(history_input_emb + [current_input_emb], dim=1)
                    history_input_emb.append(current_input_emb)
                    
                    ans = self(full_input_emb, cache=None)
                gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
                # Get COT classification for current step
            # print('ans', ans.keys())
            # if 'cot_classification_logits' in ans:
                # if self.global_rank == 0:
                #     print(t)
                #     print(T)
                #     print(gen_cot_classification.shape)
                #     print(loss_mask.shape)
                # if self.global_rank == 0:
                #     print('if_force_cot_label', if_force_cot_label)
                #     print('loss_mask', loss_mask.shape)
                #     print('ans["cot_classification_logits"]', ans["cot_classification_logits"])
            if loss_mask is not None and self.if_force_cot_label:
                gen_cot_classification[:, t] = (loss_mask[:, t]).long()
                gen_cot_logits[:, t] = ans["cot_classification_logits"][:, -1]
            else:
                gen_cot_classification[:, t] = (torch.sigmoid(ans["cot_classification_logits"][:, -1, 0]) > 0.5).long()
                gen_cot_logits[:, t] = ans["cot_classification_logits"][:, -1]

            # ===== Step 3: Sequential Semantic Prediction =====
            # Key: Use the just-predicted text[t] to predict semantic[t]
            # Get text embedding for just-predicted text token
            current_text_token = gen_text[:, t]  # (B,)
            current_text_emb = self.embed_tokens(current_text_token).unsqueeze(1)  # (B, 1, D)
            
            # Get current llm_hidden
            current_llm_hidden = ans["llm_hidden"][:, -1:, :]  # (B, 1, D)
            
            if self._use_transformer_semantic:  # False
                # TransformerSemanticPredictor: needs full history for cross-attention
                # Store current llm_hidden in history
                llm_hidden_history[:, t:t+1, :] = current_llm_hidden
                
                # Get all generated text tokens up to and including current
                all_text_tokens = gen_text[:, :t+1]  # (B, t+1)
                full_text_embeds = self.embed_tokens(all_text_tokens)  # (B, t+1, D)
                
                # Get llm_hidden history up to current step
                full_llm_hidden = llm_hidden_history[:, :t+1, :]  # (B, t+1, D)
                
                # Past semantic tokens (0 to t-1), None for first step
                past_semantic = gen_semantic[:, :t] if t > 0 else None  # (B, t) or None
                
                # Generate next semantic token with transformer
                gen_semantic[:, t] = self.semantic_predictor.generate(
                    llm_hidden=full_llm_hidden,
                    text_embed=full_text_embeds,
                    temperature=self.cfg.get("semantic_temperature", 0.9),
                    topk=self.cfg.get("semantic_topk", 20),
                    past_semantic_tokens=past_semantic,
                )  # (B,)
            else:
                # MLP SemanticTokenPredictor: frame-wise prediction
                gen_semantic[:, t] = self.semantic_predictor.generate(
                    llm_hidden=current_llm_hidden,  # (B, 1, D)
                    text_embed=current_text_emb,    # (B, 1, D)
                    temperature=self.cfg.get("semantic_temperature", 0.9),
                    topk=self.cfg.get("semantic_topk", 20),
                )  # (B,)

        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_semantic = gen_semantic[:, :T_local]
            gen_cot_classification = gen_cot_classification[:, :T_local]
            if loss_mask is not None:
                loss_mask = loss_mask[:, :T_local]

        # gen_text_strs = tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id, remove_special_tokens = False)
        if self.global_rank == 0:
            # print(gen_text_strs[0])
            # print('gen_cot_classification', gen_cot_classification)
            # print('loss_mask', loss_mask)
            min_length = min(gen_cot_classification.shape[1], loss_mask.shape[1])
            # print(loss_mask[:, :min_length] == gen_cot_classification[:, :min_length])
            print('equal_ratio', torch.sum((loss_mask[:, :min_length] == gen_cot_classification[:, :min_length]).int(), dim = -1) / min_length)
            print('loss_mask_1_ratio', torch.sum((loss_mask[:, :min_length]==1).int(), dim = -1) / min_length)
            print('gen_cot_classification_1_ratio', torch.sum((gen_cot_classification[:, :min_length]==1).int(), dim = -1) / min_length)

            cot_cls_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    gen_cot_logits[:, :min_length].squeeze(-1),  # (B, T) -> (*)
                    loss_mask[:, :min_length].float(),
                    reduction="none",
                )
            ).sum(-1) / (min_length)
            print('cot_cls_loss', cot_cls_loss)

        # gen_res_strs = []
        # gen_cot_strs = []
        # for gen_text_str in gen_text_strs:
        #     get_res_str = _extract_segments_stack_no_inner_specials(gen_text_str, "<|im_start|>", "<|im_end|>")
        #     get_cot_str = gen_text_str
        #     for res_str in get_res_str:
        #         get_cot_str = get_cot_str.replace(res_str, '')
        #     get_cot_str = get_cot_str.replace("<|im_start|>", "")
        #     get_cot_str = get_cot_str.replace("<|im_end|>", "")
            
        #     # Join list of response segments into a single string
        #     get_res_str = ' '.join(get_res_str)
            
        #     # get_cot_str = _extract_segments_stack_no_inner_specials(gen_text_str, "<|cot_start|>", "<|cot_end|>")
        #     # get_res_str = gen_text_str
        #     # for cot_str in get_cot_str:
        #     #     get_res_str = get_res_str.replace(cot_str, '')
        #     # # get_cot_str = get_cot_str.replace("<|im_start|>", "")
        #     # # get_cot_str = get_cot_str.replace("<|im_end|>", "")
            
        #     # # get_cot_str, get_res_str = extract_all_cots_and_resps(gen_text_str)
        #     # # get_cot_str = ' '.join(get_cot_str)
        #     # get_cot_str = ' '.join(get_cot_str)
        #     if get_cot_str:
        #         get_cot_str = strip_special_tokens_from_string(self.tokenizer, get_cot_str)
        #     if get_res_str:
        #         get_res_str = strip_special_tokens_from_string(self.tokenizer, get_res_str)
        #     gen_cot_strs.append(get_cot_str)
        #     gen_res_strs.append(get_res_str)
        gen_text_v2 = tokens_to_str_extract(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id,
                                  user_bos_id=self.text_bos_id, 
                                  cotstart_id=self.cotstart_id,
                                  cotend_id=self.cotend_id,
                                  eval_text_turn_taking=True)
        if self.global_rank == 0:
            print('gen_text_v2', gen_text_v2)
        gen_text_ori = tokens_to_str_ori(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id, remove_special_tokens = False)
        if self.global_rank == 0:
            print('gen_text_ori', gen_text_ori)
        # gen_text_v2_filtered = tokens_to_str_extract(gen_text * gen_cot_classification, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id,
        #                           user_bos_id=self.text_bos_id, 
        #                           cotstart_id=self.cotstart_id,
        #                           cotend_id=self.cotend_id,
        #                           eval_text_turn_taking=True)
        # if self.global_rank == 0:
        #     print('gen_text_v2_filtered', gen_text_v2_filtered)
        ans = {
            "text": gen_text_v2,
            "tokens_text": gen_text,
            "tokens_semantic": gen_semantic,  # Always return semantic tokens
            "tokens_len": lengths,
            "source_audio": input_signal,
            "source_audio_len": input_signal_lens,
        }

        # ========== Decode Semantic Tokens to Waveform ==========
        # Decode semantic tokens to waveform if requested
        if decode_audio and self.cfg.get("pretrained_flow", None):
            self.load_flow_decoder()
            with fp32_precision(), torch.no_grad():
                this_uuid = str(uuid.uuid4())
                
                # Prepare empty prompts (no speaker conditioning for semantic tokens)
                prompt_speech_feat = torch.zeros(B, 0, 80).to(self.device)
                flow_prompt_speech_token = torch.zeros(B, 0, dtype=torch.int64).to(self.device)
                spk_emb = torch.zeros(B, 192).to(self.device)
                
                # Clean up semantic tokens: replace control tokens with 0 (if any)
                # gen_semantic: (B, T) - already single-layer, no need to select codebook
                flow_input_token = gen_semantic.clone()  # (B, T)
                flow_input_token[flow_input_token > 16383] = 0  # Replace any invalid tokens
                
                # Decode semantic tokens to waveform
                # Note: WhisperVQ decoder expects semantic tokens (first codebook)
                response_speech, _ = self.audio_decoder.token2wav(
                    flow_input_token,
                    uuid=this_uuid,
                    prompt_token=flow_prompt_speech_token.to(self.device),
                    prompt_feat=prompt_speech_feat.to(self.device),
                    embedding=spk_emb,
                    finalize=True
                )
                
                ans["audio"] = response_speech  # (B, wav_len)
                ans["audio_len"] = torch.tensor([response_speech.shape[1]]).repeat(B).to(self.device)

        return ans

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn

                # Get values from model config instead of attention layer (for different model compatibility)
                try:
                    config = self.llm.config

                    # Get config values
                    num_attention_heads = getattr(config, 'num_attention_heads', None)
                    num_key_value_heads = getattr(config, 'num_key_value_heads', None)
                    hidden_size = getattr(config, 'hidden_size', None)

                    if all([num_attention_heads, num_key_value_heads, hidden_size]):
                        # Check divisibility constraints
                        for attr_name, val in [("num_attention_heads", num_attention_heads),
                                               ("num_key_value_heads", num_key_value_heads),
                                               ("hidden_size", hidden_size)]:
                            if val % tp_mesh.size() != 0:
                                logging.warning(
                                    f"config.{attr_name}={val} is not divisible by {tp_mesh.size()=}: "
                                    f"set a different tensor parallelism size to avoid errors."
                                )

                        # Set sharded values if attributes exist on attention layer
                        if hasattr(attn_layer, 'num_heads'):
                            attn_layer.num_heads = num_attention_heads // tp_mesh.size()
                        elif hasattr(attn_layer, 'num_attention_heads'):
                            attn_layer.num_attention_heads = num_attention_heads // tp_mesh.size()

                        if hasattr(attn_layer, 'num_key_value_heads'):
                            attn_layer.num_key_value_heads = num_key_value_heads // tp_mesh.size()

                        if hasattr(attn_layer, 'hidden_size'):
                            attn_layer.hidden_size = hidden_size // tp_mesh.size()

                        logging.info(f"Configured tensor parallel for attention: "
                                     f"heads={num_attention_heads // tp_mesh.size()}, "
                                     f"kv_heads={num_key_value_heads // tp_mesh.size()}, "
                                     f"hidden_size={hidden_size // tp_mesh.size()}")
                    else:
                        raise AttributeError("Required config attributes not found")

                except Exception as e:
                    logging.warning(f"Failed to configure tensor parallel using config: {e}")
                    logging.warning("Falling back to attention layer attributes...")

                    # Fallback: try original method using attention layer attributes
                    try:
                        for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                            if hasattr(attn_layer, attr):
                                val = getattr(attn_layer, attr)
                                if val % tp_mesh.size() != 0:
                                    logging.warning(
                                        f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                                        f"set a different tensor parallelism size to avoid errors."
                                    )
                                setattr(attn_layer, attr, val // tp_mesh.size())
                    except Exception as fallback_e:
                        logging.warning(f"Both config and fallback methods failed: {fallback_e}")
                        logging.warning("Skipping tensor parallel configuration for this attention layer")

            for m in (self.lm_head,):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            # Wrap gated_fusion first if it exists
            if self.gated_fusion is not None:
                self.gated_fusion = fully_shard(self.gated_fusion, **fsdp_config)
            
            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            self.whispervq = fully_shard(self.whispervq, **fsdp_config)
            self.cot_classification_head = fully_shard(self.cot_classification_head, **fsdp_config)
            
            # Wrap semantic prediction module
            self.semantic_predictor = fully_shard(self.semantic_predictor, **fsdp_config)

            self.non_causal_encoder = fully_shard(self.non_causal_encoder, **fsdp_config)
            if self.non_causal_proj_in is not None:
                self.non_causal_proj_in = fully_shard(self.non_causal_proj_in, **fsdp_config)
            if self.non_causal_proj_out is not None:
                self.non_causal_proj_out = fully_shard(self.non_causal_proj_out, **fsdp_config)
                
            # self.z_projection = fully_shard(self.z_projection, **fsdp_config)
            self.z_head = fully_shard(self.z_head, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)

import re
from typing import List, Tuple
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>|]*\|>")  # 任意特殊标记 <|...|>

def _extract_segments_stack_no_inner_specials(
    text: str,
    start_token: str,
    end_token: str,
) -> List[str]:
    if not isinstance(text, str) or not text:
        return []

    token_re = re.compile(f"(?:{re.escape(start_token)}|{re.escape(end_token)})", re.DOTALL)
    segments: List[str] = []
    stack: List[int] = []  # 内容起点（start_token 之后的索引）

    for m in token_re.finditer(text):
        tok = m.group(0)
        if tok == start_token:
            stack.append(m.end())
        else:  # end_token
            if stack:
                start_idx = stack.pop()
                content = text[start_idx:m.start()]
                # 内部不得包含任意特殊标记
                if not _SPECIAL_TOKEN_RE.search(content):
                    segments.append(content.strip())
    
    # Handle unclosed segments (start_token without end_token at the end of string)
    if stack:
        # Take the last unclosed start_token (innermost)
        start_idx = stack[-1]
        content = text[start_idx:]
        # Ensure no inner special tokens (except potentially whitespace)
        if not _SPECIAL_TOKEN_RE.search(content):
            segments.append(content.strip())
            
    return segments

def extract_all_cots_and_resps(text: str) -> Tuple[List[str], List[str]]:
    """
    - CoT 段：由 <|cot_start|> 与 <|cot_end|> 成对包围，且内部不得含任意 <|...|>；允许空段
    - Resp 段：由 <|im_start|> 与 <|im_end|> 成对包围，且内部不得含任意 <|...|>；允许空段
    返回 (all_cots, all_resps)
    """
    cot_start_token = "<|cot_start|>"
    cot_end_token = "<|cot_end|>"
    bos_token = "<|im_start|>"
    eos_token = "<|im_end|>"

    cots = _extract_segments_stack_no_inner_specials(text, cot_start_token, cot_end_token)
    resps = _extract_segments_stack_no_inner_specials(text, bos_token, eos_token)
    return cots, resps

def strip_special_tokens_from_string(tokenizer, text: str) -> str:
    # tokenizer.tokenizer.all_special_tokens 是所有特殊符号的字符串列表
    specials = sorted(tokenizer.tokenizer.all_special_tokens, key=len, reverse=True)
    if not specials:
        return text
    pattern = re.compile("|".join(re.escape(s) for s in specials))
    # 去掉多余空白
    return re.sub(r"\s+", " ", pattern.sub("", text)).strip()

def tokens_to_str_ori(tokens: torch.Tensor, lengths: torch.Tensor, tokenizer: AutoTokenizer, pad_id: int, remove_special_tokens = True) -> list[str]:
    ans = []
    for hyp_ids, hyp_len in zip(tokens.cpu(), lengths.cpu()):
        hyp_ids = hyp_ids[:hyp_len]
        hyp_ids = hyp_ids[hyp_ids != pad_id]
        ans.append(tokenizer.ids_to_text(hyp_ids, remove_special_tokens))
    return ans