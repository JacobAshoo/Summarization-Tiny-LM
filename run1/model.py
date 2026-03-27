"""
Seq2Seq Transformer for Text Summarization
==========================================
Architecture:
  - Encoder-Decoder Transformer (~200M parameters)
  - Standard Multi-Head Self-Attention with RoPE
  - RoPE (Rotary Positional Embeddings) via torchtune
  - SwiGLU Feed-Forward Network
  - Pre-RMSNorm (applied before each sub-layer)
  - Flash Attention via PyTorch scaled_dot_product_attention
  - No bias terms anywhere
  - No weight tying (encoder embed, decoder embed, output proj are all separate)

Dependencies:
  pip install torch torchtune
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchtune.modules import RotaryPositionalEmbeddings


# ── Model Dimensions ───────────────────────────────────────────────────────────
D_MODEL    = 512# was 512
ENC_LAYERS = 6     # was 4
DEC_LAYERS = 6     # was 4
FFN_HIDDEN = 4096  # was 1365 — scale with d_model
NUM_HEADS  = 8    # was 8 — keep HEAD_DIM = 64
HEAD_DIM      = D_MODEL // NUM_HEADS

# ── Vocabulary & Sequence ──────────────────────────────────────────────────────
VOCAB_SIZE    = 50_000
MAX_SRC_LEN   = 2002
MAX_TGT_LEN   = 1002

# ── Regularization ─────────────────────────────────────────────────────────────
DROPOUT       = 0

# ── Misc ───────────────────────────────────────────────────────────────────────
EPS           = 1e-6    # RMSNorm epsilon
ROPE_BASE     = 10000   # RoPE theta base


# ──────────────────────────────────────────────────────────────────────────────
# COMPONENT 1: RMSNorm
# Using PyTorch built-in (added in PyTorch 2.4)
# nn.RMSNorm(D_MODEL, eps=EPS) — no bias, just learned scale γ
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# COMPONENT 2: RoPE
# Using torchtune built-in RotaryPositionalEmbeddings
# Applied inside attention on Q and K after projection, before dot product
# Not used in cross-attention (Q and K live in different sequences)
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# COMPONENT 3: SwiGLU Feed-Forward Network
# ──────────────────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Two parallel projections (gate + up) followed by a down projection.
    Gate is passed through SiLU (Swish) then element-wise multiplied with up.
    No bias anywhere.

    FFN(x) = down( SiLU(gate(x)) * up(x) )
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D_MODEL, FFN_HIDDEN, bias=False)
        self.up   = nn.Linear(D_MODEL, FFN_HIDDEN, bias=False)
        self.down = nn.Linear(FFN_HIDDEN, D_MODEL, bias=False)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_MODEL)
        return self.down(self.act(self.gate(x)) * self.up(x))


# ──────────────────────────────────────────────────────────────────────────────
# COMPONENT 4a: Self-Attention
# ──────────────────────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention with RoPE.

    Projects Q, K, V, applies RoPE to Q and K, then runs
    PyTorch scaled_dot_product_attention (FlashAttention kernel when available).

    is_causal=True  → decoder self-attention (causal mask applied inside SDPA)
    is_causal=False → encoder self-attention (bidirectional, no mask)
    """
    def __init__(self, is_causal: bool = False):
        super().__init__()
        self.is_causal = is_causal
        max_len = MAX_TGT_LEN if is_causal else MAX_SRC_LEN

        self.q_proj   = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj   = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj   = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

        self.rope = RotaryPositionalEmbeddings(
            dim=HEAD_DIM,
            max_seq_len=max_len,
            base=ROPE_BASE,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_MODEL)
        B, T, _ = x.shape

        # ── 1. Project and reshape to (B, T, H, D) ────────────────
        q = self.q_proj(x).view(B, T, NUM_HEADS, HEAD_DIM)
        k = self.k_proj(x).view(B, T, NUM_HEADS, HEAD_DIM)
        v = self.v_proj(x).view(B, T, NUM_HEADS, HEAD_DIM)

        # ── 2. Apply RoPE to Q and K ───────────────────────────────
        q = self.rope(q)
        k = self.rope(k)

        # ── 3. Transpose to (B, H, T, D) for SDPA ─────────────────
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── 4. Scaled dot-product attention ───────────────────────
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=self.is_causal,
        )  # (B, H, T, D)

        # ── 5. Reshape and project ─────────────────────────────────
        out = out.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# COMPONENT 4b: Cross-Attention
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Cross-Attention for the Decoder.

    Q comes from the decoder, K and V come from the encoder output.
    No RoPE — Q and K live in different sequences with different position
    indices, so relative rotation would be meaningless.
    Always non-causal.
    """
    def __init__(self):
        super().__init__()
        self.q_proj   = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj   = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj   = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x:       (B, T, D_MODEL) — from decoder (query source)
        # context: (B, S, D_MODEL) — from encoder (key/value source)
        B, T, _ = x.shape
        S        = context.shape[1]

        q = self.q_proj(x).view(B, T, NUM_HEADS, HEAD_DIM)
        k = self.k_proj(context).view(B, S, NUM_HEADS, HEAD_DIM)
        v = self.v_proj(context).view(B, S, NUM_HEADS, HEAD_DIM)

        # SDPA expects (B, H, S, D)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, H, S, D)
        v = v.transpose(1, 2)  # (B, H, S, D)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=False,
        )  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 1: Encoder Block
# ──────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block.

    Pre-RMSNorm + residual pattern:
        x = x + SelfAttn( RMSNorm(x) )
        x = x + FFN(      RMSNorm(x) )

    norm1 and norm2 are separate instances — each learns its own γ vector.
    is_causal=False — encoder sees the full source sequence bidirectionally.
    """
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(D_MODEL, eps=EPS)
        self.attn  = SelfAttention(is_causal=False)
        self.norm2 = nn.RMSNorm(D_MODEL, eps=EPS)
        self.ffn   = SwiGLUFFN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D_MODEL)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn( self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 2: Decoder Block
# ──────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block.

    Pre-RMSNorm + residual pattern, three sub-layers:
        x = x + CausalSelfAttn( RMSNorm(x) )
        x = x + CrossAttn(      RMSNorm(x), enc_out )
        x = x + FFN(            RMSNorm(x) )

    enc_out is passed through unchanged — computed once by the encoder,
    reused at every decoder block and every decoding step.
    """
    def __init__(self):
        super().__init__()
        self.norm1      = nn.RMSNorm(D_MODEL, eps=EPS)
        self.self_attn  = SelfAttention(is_causal=True)
        self.norm2      = nn.RMSNorm(D_MODEL, eps=EPS)
        self.cross_attn = CrossAttention()
        self.norm3      = nn.RMSNorm(D_MODEL, eps=EPS)
        self.ffn        = SwiGLUFFN()

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        # x:       (B, T, D_MODEL) — target sequence
        # enc_out: (B, S, D_MODEL) — encoder output
        x = x + self.self_attn( self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), enc_out)
        x = x + self.ffn(       self.norm3(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# STACK 1: Encoder
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Full Encoder Stack.

    Embedding → ENC_LAYERS x EncoderBlock → RMSNorm

    gradient_checkpointing: recomputes activations during backward instead of
    storing them — reduces memory at ~33% extra compute cost.
    """
    def __init__(self, gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.embed  = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.blocks = nn.ModuleList([EncoderBlock() for _ in range(ENC_LAYERS)])
        self.norm   = nn.RMSNorm(D_MODEL, eps=EPS)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (B, S) — source token ids
        x = self.embed(src)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)  # (B, S, D_MODEL)


# ──────────────────────────────────────────────────────────────────────────────
# STACK 2: Decoder
# ──────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Full Decoder Stack.

    Embedding → DEC_LAYERS x DecoderBlock → RMSNorm

    enc_out is passed into every block — computed once by the encoder
    and reused across all layers and all decoding steps.
    """
    def __init__(self, gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.embed  = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(DEC_LAYERS)])
        self.norm   = nn.RMSNorm(D_MODEL, eps=EPS)

    def forward(self, tgt: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        # tgt:     (B, T) — target token ids
        # enc_out: (B, S, D_MODEL) — from encoder
        x = self.embed(tgt)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, enc_out, use_reentrant=False)
            else:
                x = block(x, enc_out)
        return self.norm(x)  # (B, T, D_MODEL)


# ──────────────────────────────────────────────────────────────────────────────
# FULL MODEL: Seq2Seq Transformer
# ──────────────────────────────────────────────────────────────────────────────

class Seq2SeqTransformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for Text Summarization (~200M parameters).

    Components:
      Encoder:           src tokens  → contextual representations
      Decoder:           tgt tokens + enc_out → hidden states
      Output projection: hidden states → vocab logits

    No weight tying — encoder embedding, decoder embedding, and output
    projection are all separate learned matrices.
    No bias anywhere.

    Training (teacher forcing):
        logits = model(src, tgt_shifted_right)
        loss   = cross_entropy(logits, tgt)

    Inference (autoregressive):
        enc_out = model.encode(src)
        for each step:
            logits  = model.decode_step(tgt_so_far, enc_out)
            next_id = logits[:, -1].argmax(dim=-1)
    """
    def __init__(self, gradient_checkpointing: bool = False):
        super().__init__()
        self.encoder  = Encoder(gradient_checkpointing)
        self.decoder  = Decoder(gradient_checkpointing)
        self.out_proj = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src: (B, S) — source token ids
        # tgt: (B, T) — target token ids, shifted right (teacher forcing)
        enc_out = self.encoder(src)           # (B, S, D_MODEL)
        dec_out = self.decoder(tgt, enc_out)  # (B, T, D_MODEL)
        return self.out_proj(dec_out)         # (B, T, VOCAB_SIZE)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode source once. Reuse enc_out during autoregressive decoding."""
        return self.encoder(src)  # (B, S, D_MODEL)

    def decode_step(self, tgt: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        """Single decode step. Called token-by-token during inference."""
        dec_out = self.decoder(tgt, enc_out)  # (B, T, D_MODEL)
        return self.out_proj(dec_out)         # (B, T, VOCAB_SIZE)


# ──────────────────────────────────────────────────────────────────────────────
# PARAMETER COUNT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = Seq2SeqTransformer()

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total / 1e6:.1f}M\n")

    for name, module in [("Encoder",           model.encoder),
                         ("Decoder",           model.decoder),
                         ("Output Projection", model.out_proj)]:
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params / 1e6:.1f}M")
