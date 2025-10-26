# model_wrapper.py - Transformer Chatbot Model Wrapper
# Updated to match your training code architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import json
import re
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from dataclasses import dataclass
import sentencepiece as spm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class TrainingConfig:
    """Model configuration matching your training code"""
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 50
    vocab_size: int = 16000
    use_relative_position: bool = True


# ============================================
# TEXT PROCESSOR
# ============================================

class UrduTextProcessor:
    """Urdu-specific text preprocessing matching your training code"""
    
    def __init__(self):
        self.diacritics = [
            '\u064B', '\u064C', '\u064D', '\u064E', '\u064F',
            '\u0650', '\u0651', '\u0652', '\u0653', '\u0654',
            '\u0655', '\u0656', '\u0657', '\u0658', '\u0670'
        ]
        
        self.normalization_map = {
            'آ': 'ا', 'أ': 'ا', 'إ': 'ا',
            'ۀ': 'ہ', 'ة': 'ہ',
            'ي': 'ی', 'ك': 'ک',
        }
    
    def remove_diacritics(self, text: str) -> str:
        for diacritic in self.diacritics:
            text = text.replace(diacritic, '')
        return text
    
    def normalize_characters(self, text: str) -> str:
        for old_char, new_char in self.normalization_map.items():
            text = text.replace(old_char, new_char)
        return text
    
    def clean_text(self, text: str) -> str:
        text = ' '.join(text.split())
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\s۔،؟!]', '', text)
        text = self.normalize_characters(text)
        text = self.remove_diacritics(text)
        text = ' '.join(text.split())
        return text.strip()
    
    def preprocess(self, text: str) -> str:
        return self.clean_text(text)


# ============================================
# TOKENIZER
# ============================================

class UrduSentencePieceTokenizer:
    """SentencePiece tokenizer matching your training code"""
    
    def __init__(self, model_prefix: str = 'urdu_sp'):
        self.model_prefix = model_prefix
        self.sp = None
        self.vocab_size = 0
    
    def load(self, model_path: str) -> bool:
        """Load pre-trained SentencePiece model"""
        try:
            if os.path.exists(model_path):
                self.sp = spm.SentencePieceProcessor()
                self.sp.load(model_path)
                self.vocab_size = self.sp.get_piece_size()
                print(f"✓ Loaded tokenizer: {model_path}")
                print(f"  Vocab size: {self.vocab_size}")
                print(f"  PAD: {self.sp.pad_id()}, UNK: {self.sp.unk_id()}")
                print(f"  BOS: {self.sp.bos_id()}, EOS: {self.sp.eos_id()}")
                return True
            else:
                print(f"✗ Tokenizer file not found: {model_path}")
                return False
        except Exception as e:
            print(f"✗ Error loading tokenizer: {e}")
            return False
    
    def encode(self, text: str, add_special_tokens: bool = True, max_length: int = None) -> List[int]:
        """Encode text to token IDs"""
        if not self.sp:
            raise ValueError("Tokenizer not loaded")
        
        ids = self.sp.encode(text, out_type=int)
        
        if add_special_tokens:
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        
        if max_length:
            if len(ids) < max_length:
                ids = ids + [self.sp.pad_id()] * (max_length - len(ids))
            else:
                ids = ids[:max_length-1] + ([self.sp.eos_id()] if add_special_tokens else [ids[:max_length]])
        
        return ids
    
    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if not self.sp:
            raise ValueError("Tokenizer not loaded")
        
        special_ids = {self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id(), self.sp.unk_id()}
        
        if remove_special_tokens:
            ids = [id for id in ids if id not in special_ids]
        
        return self.sp.decode(ids).strip()
    
    def get_vocab_size(self) -> int:
        return self.vocab_size if self.sp else 0


# ============================================
# MODEL ARCHITECTURE
# ============================================

class PositionalEncoding(nn.Module):
    """Absolute positional encoding"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(1))
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class RelativePositionalBias(nn.Module):
    """Relative position bias for attention (T5-style)"""
    
    def __init__(self, n_heads: int, max_distance: int = 32):
        super().__init__()
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(2 * max_distance + 1, n_heads)
    
    def forward(self, Q_len: int, K_len: int, device: torch.device):
        Q_positions = torch.arange(Q_len, device=device)
        K_positions = torch.arange(K_len, device=device)
        
        relative_positions = Q_positions.unsqueeze(1) - K_positions.unsqueeze(0)
        relative_positions = torch.clamp(relative_positions, -self.max_distance, self.max_distance)
        relative_positions = relative_positions + self.max_distance
        
        bias = self.relative_attention_bias(relative_positions)
        return bias.permute(2, 0, 1).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional relative position bias"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_relative_position: bool = False, max_distance: int = 32):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
        
        self.use_relative_position = use_relative_position
        if use_relative_position:
            self.relative_bias = RelativePositionalBias(n_heads, max_distance)
    
    def forward(self, Q, K, V, mask=None):
        Q_len, batch_size, _ = Q.size()
        K_len, _, _ = K.size()
        
        Q = self.W_q(Q).view(Q_len, batch_size, self.n_heads, self.d_k).permute(1, 2, 0, 3)
        K = self.W_k(K).view(K_len, batch_size, self.n_heads, self.d_k).permute(1, 2, 0, 3)
        V = self.W_v(V).view(K_len, batch_size, self.n_heads, self.d_k).permute(1, 2, 0, 3)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if self.use_relative_position:
            rel_bias = self.relative_bias(Q_len, K_len, Q.device)
            scores = scores + rel_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        self.attention_weights = attention_weights.detach()
        
        context = torch.matmul(attention_weights, V)
        context = context.permute(2, 0, 1, 3).contiguous().view(Q_len, batch_size, self.d_model)
        
        return self.W_o(context)


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder layer"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_relative_position: bool = False):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout, use_relative_position)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer with cross-attention"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_relative_position: bool = False):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout, use_relative_position)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout, use_relative_position)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_out, tgt_mask=None, src_mask=None):
        attn_out = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        cross_attn_out = self.cross_attention(x, encoder_out, encoder_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder stack"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1, use_relative_position: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, use_relative_position)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder stack"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1, use_relative_position: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, use_relative_position)
            for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_out, tgt_mask=None, src_mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_out, tgt_mask, src_mask)
        return self.output_projection(x)


class UrduTransformerChatbot(nn.Module):
    """Complete Transformer chatbot matching your training architecture"""
    
    def __init__(self, vocab_size: int, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.pad_idx = 0
        self.vocab_size = vocab_size
        
        self.encoder = TransformerEncoder(
            vocab_size, config.d_model, config.n_heads, config.n_encoder_layers,
            config.d_ff, config.dropout, config.use_relative_position
        )
        
        self.decoder = TransformerDecoder(
            vocab_size, config.d_model, config.n_heads, config.n_decoder_layers,
            config.d_ff, config.dropout, config.use_relative_position
        )
    
    def create_src_mask(self, src):
        """Mask for padding in encoder"""
        mask = (src != self.pad_idx).transpose(0, 1)
        return mask.unsqueeze(1).unsqueeze(1)
    
    def create_tgt_mask(self, tgt):
        """Causal + padding mask for decoder"""
        tgt_len = tgt.size(0)
        batch_size = tgt.size(1)
        
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1)
        causal_mask = (causal_mask == 0)
        
        pad_mask = (tgt != self.pad_idx).transpose(0, 1)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
        pad_mask = pad_mask.expand(batch_size, 1, tgt_len, tgt_len)
        
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = causal_mask & pad_mask
        
        return attention_mask
    
    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, tgt_mask, src_mask)
        
        return decoder_out
    
    def generate(self, src, max_length=50, temperature=1.0, top_k=0, top_p=0.9):
        """Generate response with nucleus sampling"""
        self.eval()
        
        with torch.no_grad():
            src_mask = self.create_src_mask(src)
            encoder_out = self.encoder(src, src_mask)
            
            batch_size = src.size(1)
            eos_id = 3
            bos_id = 2
            
            generated = torch.full((1, batch_size), bos_id, dtype=torch.long, device=src.device)
            
            for step in range(max_length - 1):
                tgt_mask = self.create_tgt_mask(generated)
                output = self.decoder(generated, encoder_out, tgt_mask, src_mask)
                
                next_logits = output[-1, :, :] / temperature
                next_logits = next_logits[:, :self.vocab_size]
                
                if top_k > 0:
                    top_k = min(top_k, next_logits.size(-1))
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = -float('Inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(next_logits, dim=-1)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.clamp(next_token, 0, self.vocab_size - 1)
                
                generated = torch.cat([generated, next_token.transpose(0, 1)], dim=0)
                
                if (next_token == eos_id).all():
                    break
            
            return generated


# ============================================
# INFERENCE CLASS
# ============================================

class TransformerChatbotInference:
    """Main chatbot inference class"""
    
    def __init__(self, model_path: str = 'best_xlarge_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.processor = UrduTextProcessor()
        self.config = None
        self.session_stats = {
            'total_conversations': 0,
            'avg_response_time': 0,
            'total_tokens_processed': 0
        }
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load the trained model and tokenizer"""
        try:
            print(f"\n{'='*60}")
            print(f"LOADING URDU TRANSFORMER CHATBOT")
            print(f"{'='*60}")
            print(f"Model path: {model_path}")
            print(f"Device: {self.device}")
            
            # Load tokenizer first
            self.tokenizer = UrduSentencePieceTokenizer()
            tokenizer_path = 'urdu_sp.model'
            
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
            
            if not self.tokenizer.load(tokenizer_path):
                raise ValueError("Failed to load tokenizer")
            
            # Load model checkpoint
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"\nLoading checkpoint: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract config and state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    config_dict = checkpoint.get('config', {})
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    config_dict = checkpoint.get('config', {})
                else:
                    # Assume it's a plain state dict
                    state_dict = checkpoint
                    config_dict = {}
            else:
                raise ValueError("Invalid checkpoint format")
            
            # Create config from checkpoint or use defaults
            if config_dict:
                self.config = TrainingConfig(
                    d_model=config_dict.get('d_model', 768),
                    n_heads=config_dict.get('n_heads', 12),
                    n_encoder_layers=config_dict.get('n_encoder_layers', 6),
                    n_decoder_layers=config_dict.get('n_decoder_layers', 6),
                    d_ff=config_dict.get('d_ff', 3072),
                    dropout=config_dict.get('dropout', 0.1),
                    max_seq_length=config_dict.get('max_seq_length', 50),
                    vocab_size=self.tokenizer.get_vocab_size(),
                    use_relative_position=config_dict.get('use_relative_position', True)
                )
            else:
                # Default xlarge config
                self.config = TrainingConfig(
                    d_model=768,
                    n_heads=12,
                    n_encoder_layers=6,
                    n_decoder_layers=6,
                    d_ff=3072,
                    dropout=0.1,
                    max_seq_length=50,
                    vocab_size=self.tokenizer.get_vocab_size(),
                    use_relative_position=True
                )
            
            print(f"\nModel Configuration:")
            print(f"  d_model: {self.config.d_model}")
            print(f"  n_heads: {self.config.n_heads}")
            print(f"  Encoder layers: {self.config.n_encoder_layers}")
            print(f"  Decoder layers: {self.config.n_decoder_layers}")
            print(f"  d_ff: {self.config.d_ff}")
            print(f"  Vocab size: {self.config.vocab_size}")
            
            # Create model
            self.model = UrduTransformerChatbot(
                vocab_size=self.config.vocab_size,
                config=self.config
            ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\n✓ Model loaded successfully!")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Device: {self.device}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def chat(self, input_text: str, max_length: int = 50, temperature: float = 0.8,
             top_p: float = 0.9) -> Tuple[str, float]:
        """Generate chatbot response"""
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded", 0
        
        start_time = time.time()
        
        try:
            # Preprocess input
            cleaned_text = self.processor.preprocess(input_text.strip())
            if not cleaned_text:
                return "معذرت، میں خالی متن کو سمجھ نہیں سکتا", time.time() - start_time
            
            # Encode input
            src_ids = self.tokenizer.encode(
                cleaned_text, 
                add_special_tokens=True, 
                max_length=self.config.max_seq_length
            )
            src = torch.tensor([src_ids]).transpose(0, 1).to(self.device)
            
            # Generate response
            self.model.eval()
            with torch.no_grad():
                generated = self.model.generate(
                    src,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p
                )
            
            # Decode response
            response_ids = generated[:, 0].cpu().tolist()
            response_text = self.tokenizer.decode(response_ids, remove_special_tokens=True).strip()
            
            if not response_text:
                response_text = "معذرت، میں اس وقت جواب نہیں دے سکتا"
            
            response_time = time.time() - start_time
            self._update_stats(input_text, response_time)
            
            return response_text, response_time
            
        except Exception as e:
            print(f"Chat error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", time.time() - start_time
    
    def _update_stats(self, input_text: str, response_time: float):
        """Update session statistics"""
        self.session_stats['total_conversations'] += 1
        self.session_stats['total_tokens_processed'] += len(input_text.split())
        
        if self.session_stats['avg_response_time'] == 0:
            self.session_stats['avg_response_time'] = response_time
        else:
            n = self.session_stats['total_conversations']
            self.session_stats['avg_response_time'] = (
                (self.session_stats['avg_response_time'] * (n - 1) + response_time) / n
            )
    
    def get_stats(self) -> Dict:
        """Get session statistics"""
        return self.session_stats.copy()