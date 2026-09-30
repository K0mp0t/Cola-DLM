import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Cola-DLM Continue Pretraining")
# Model
parser.add_argument("--vae-type", type=str, choices=["vanilla", "custom"])
parser.add_argument("--dit-path", type=str, default="hf_models/cola_dlm/cola_dit")
parser.add_argument("--vae-path", type=str, default="hf_models/cola_dlm/cola_vae")
parser.add_argument("--tokenizer-path", type=str, default="hf_models/tokenizer.json")
# Output
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--max-seq-len", type=int, default=512)
# Flow matching
parser.add_argument("--timestep-dist", type=str, default="logit_normal", choices=["logit_normal", "uniform"])
parser.add_argument("--logit-normal-loc", type=float, default=0.0)
parser.add_argument("--logit-normal-scale", type=float, default=1.0)
parser.add_argument("--T", type=float, default=1000.0)
# VAE
parser.add_argument("--vae-mode", type=str, default="sample", choices=["sample", "mode"])
parser.add_argument("--block-size-probs", type=str, default=None,
                    help="Comma-separated probs for block sizes 1,2,4,...,block_size")
# Simulated prompt-response blocks
parser.add_argument("--prompt-block-prob", type=float, default=0.05,
                    help="Prob of simulating a [P,R] boundary within a noisy block")
# Eval / Save
parser.add_argument("--eval-steps", type=int, default=1000)
# Data
parser.add_argument("--data-dir", type=str, default="cache_nanochat/base_data_climbmix")
# Attention backend
parser.add_argument("--attn-backend", type=str, default="naive", choices=["naive", "sdpa", "flex"],
                    help="Attention backend: 'naive' (manual matmul), 'sdpa' (PyTorch SDPA), 'flex' (FlexAttention)")
args = parser.parse_args()


if "RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{ddp_local_rank}")
    torch.cuda.set_device(device)
else:
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
master_process = ddp_rank == 0

def print0(*a, **kw):
    if master_process:
        print(*a, **kw, flush=True)

from cola_dlm import ColaDiTModel, ColaTextVAEModel
from cola_dlm.modeling_custom_vae import CustomTextVAEModel
from cola_dlm.attention_utils import create_2l_block_causal_mask

attn_backend = args.attn_backend
if attn_backend != "naive":
    from cola_dlm.modeling_cola_dit import set_attn_backend
    set_attn_backend(attn_backend)
    print0(f"Attention backend: {attn_backend}")
if attn_backend == "flex":
    from cola_dlm.attention_utils import create_2l_flex_block_mask

print0("Loading models...")
if args.vae_type == 'vanilla':
    vae = ColaTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
elif args.vae_type == 'custom':
    vae = CustomTextVAEModel.from_pretrained(args.vae_path).to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)

dit = ColaDiTModel.from_pretrained(args.dit_path).to(device).train()

default_block_size = dit.block_size
latent_dim = vae.config.latent_dim
patch_size = vae.patch_size
T = args.T

assert patch_size == 1, f"Only patch_size=1 is supported, got {patch_size}"

# Block size config
n_candidates = int(math.log2(default_block_size)) + 1
BLOCK_SIZES = [2 ** i for i in range(n_candidates)]
if args.block_size_probs is not None:
    BLOCK_SIZE_PROBS = [float(x) for x in args.block_size_probs.split(",")]
else:
    BLOCK_SIZE_PROBS = [0.0] * (n_candidates - 1) + [1.0]
assert len(BLOCK_SIZE_PROBS) == n_candidates
assert abs(sum(BLOCK_SIZE_PROBS) - 1.0) < 1e-6

print0(f"DiT: {sum(p.numel() for p in dit.parameters()):,} params, block_size={default_block_size}")
print0(f"Training block sizes: {dict(zip(BLOCK_SIZES, BLOCK_SIZE_PROBS))}")
print0(f"VAE: {sum(p.numel() for p in vae.parameters()):,} params (frozen)")

# DDP wrap
orig_dit = dit
if ddp_world_size > 1:
    dit = torch.nn.parallel.DistributedDataParallel(dit, device_ids=[ddp_local_rank])


from cola_dlm.dataloader import pretrain_data_loader

STOP_TOKEN_ID = 47774  # ■

print0("Initializing data loader...")
data_dir = os.path.abspath(args.data_dir)
val_loader = pretrain_data_loader(
    args.tokenizer_path, data_dir, args.device_batch_size, args.max_seq_len,
    split="val", device=device,
)

def sample_block_size():
    return BLOCK_SIZES[torch.multinomial(torch.tensor(BLOCK_SIZE_PROBS), 1).item()]


# ---------------------------------------------------------------------------
# Noisy copy construction for pretraining
# ---------------------------------------------------------------------------
def build_noisy_sample_pretrain(z_0, t_val, z_1, sample_block_size):
    """All-token noisy construction with optional simulated [P,R] boundaries.

    With probability prompt_block_prob per block, a random split creates a
    clean left segment (ts=0, loss=0) and noisy right segment (ts=t*T, loss=1),
    simulating the conditioning pattern used at inference.
    """
    L = z_0.shape[0]
    z_noisy = (1 - t_val) * z_0 + t_val * z_1
    loss_mask = torch.ones(L, device=z_0.device)
    ts_noisy = torch.full((L,), t_val * T, device=z_0.device)
    target = z_1 - z_0

    if args.prompt_block_prob > 0:
        for blk_start in range(0, L, sample_block_size):
            blk_end = min(blk_start + sample_block_size, L)
            blk_len = blk_end - blk_start
            if blk_len <= 1:
                continue
            if random.random() < args.prompt_block_prob:
                split = random.randint(1, blk_len - 1)
                s = slice(blk_start, blk_start + split)
                z_noisy[s] = z_0[s]
                ts_noisy[s] = 0.0
                loss_mask[s] = 0.0

    return z_noisy, loss_mask, target, ts_noisy


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------
def sample_timestep(batch_size):
    if args.timestep_dist == "uniform":
        return torch.rand(batch_size, device=device)
    u = torch.randn(batch_size, device=device)
    return torch.sigmoid(args.logit_normal_loc + args.logit_normal_scale * u)


# ---------------------------------------------------------------------------
# Prepare batch: token IDs -> VAE latents
# ---------------------------------------------------------------------------
def prepare_batch(inputs):
    """Convert (B, T) token IDs to list of (z_0, L, block_size) tuples."""
    B = inputs.shape[0]
    batch = []
    for i in range(B):
        bs = sample_block_size()
        token_row = inputs[i]
        L = token_row.shape[0]
        pad_len = (bs - L % bs) % bs
        if pad_len > 0:
            token_row = torch.cat([
                token_row,
                torch.full((pad_len,), STOP_TOKEN_ID, device=device, dtype=torch.long),
            ])

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc = vae.encode([token_row])

        if args.vae_mode == "sample" and enc.latent_dists is not None:
            z_0 = enc.latent_dists[0].sample().float()
        else:
            z_0 = enc.latents_list[0].float()
        z_0 = (z_0 - vae.shifting_factor) * vae.scaling_factor
        batch.append((z_0, z_0.shape[0], bs))
    return batch


# ---------------------------------------------------------------------------
# Flow matching step
# ---------------------------------------------------------------------------
def flow_matching_step(dit_model, batch):
    B = len(batch)
    t = sample_timestep(B)

    extended_list = []
    target_list = []
    mask_list = []
    seq_lens = []
    block_sizes_list = []
    k_pos_list = []
    q_pos_list = []
    ts_list = []

    noisy_first = (attn_backend == "flex")

    for i, (z_0, L, bs) in enumerate(batch):
        z_1 = torch.randn_like(z_0)
        z_noisy, loss_mask, target_vel, ts_noisy = build_noisy_sample_pretrain(
            z_0, t[i].item(), z_1, bs
        )

        if noisy_first:
            extended_list.append(torch.cat([z_noisy, z_0.detach()], dim=0))
            ts_list.append(torch.cat([ts_noisy, torch.zeros(L, device=device)]))
        else:
            extended_list.append(torch.cat([z_0.detach(), z_noisy], dim=0))
            ts_list.append(torch.cat([torch.zeros(L, device=device), ts_noisy]))

        target_list.append(target_vel)
        mask_list.append(loss_mask)
        seq_lens.append(L)
        block_sizes_list.append(bs)

        positions = torch.arange(L, device=device)
        k_pos_list.append(torch.cat([positions, positions]))
        q_pos_list.append(torch.cat([positions, positions]))

    txt = torch.cat(extended_list, dim=0)
    ext_lens = [2 * sl for sl in seq_lens]
    txt_shape = torch.tensor([[el] for el in ext_lens], dtype=torch.long, device=device)
    txt_q_shape = txt_shape.clone()

    k_position_ids = torch.cat(k_pos_list, dim=0)
    q_position_ids = torch.cat(q_pos_list, dim=0)
    timestep = torch.cat(ts_list, dim=0)

    if attn_backend == "flex":
        attn_mask = create_2l_flex_block_mask(
            txt_shape, txt_q_shape,
            seq_lens=seq_lens, block_size=block_sizes_list,
            device=device,
        )
    else:
        attn_mask = create_2l_block_causal_mask(
            txt_shape, txt_q_shape,
            seq_lens=seq_lens, block_size=block_sizes_list,
            dtype=torch.bfloat16, device=device,
        )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = dit_model(
            txt=txt.to(torch.bfloat16),
            txt_shape=txt_shape,
            txt_q_shape=txt_q_shape,
            timestep=timestep.to(torch.bfloat16),
            k_position_ids=k_position_ids,
            q_position_ids=q_position_ids,
            attn_mask_override=attn_mask,
        )

    pred_list = []
    offset = 0
    for i, sl in enumerate(seq_lens):
        sample_out = out.txt_sample[offset : offset + 2 * sl]
        if noisy_first:
            pred_list.append(sample_out[:sl])
        else:
            pred_list.append(sample_out[sl:])
        offset += 2 * sl

    pred = torch.cat(pred_list, dim=0).float()
    target = torch.cat(target_list, dim=0)
    loss_mask = torch.cat(mask_list, dim=0)

    error = ((pred - target) ** 2).mean(dim=-1)
    num_masked = loss_mask.sum().clamp(min=1.0)
    loss = (error * loss_mask).sum() / num_masked

    target_energy = target.square().mean(dim=-1)
    relative_loss = ((error * loss_mask).sum() / (target_energy * loss_mask).sum().clamp_min(1e-8))

    return loss, relative_loss


@torch.no_grad()
def evaluate(dit_model, eval_steps):
    dit_model.eval()
    losses = []
    relative_losses = []
    for _ in range(eval_steps):
        inputs, _, _ = next(val_loader)
        batch = prepare_batch(inputs)
        loss, relative_loss = flow_matching_step(dit_model, batch)
        losses.append(loss.item())
        relative_losses.append(relative_loss.item())
    dit_model.train()
    avg_loss = sum(losses) / len(losses)
    avg_relative_loss = sum(relative_losses) / len(relative_losses)
    if ddp_world_size > 1:
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
        relative_loss_tensor = torch.tensor([avg_relative_loss], device=device)
        dist.all_reduce(relative_loss_tensor, op=dist.ReduceOp.AVG)
        avg_relative_loss = relative_loss_tensor.item()
    return avg_loss, avg_relative_loss

val_loss, val_relative_loss = evaluate(dit, args.eval_steps)
print0(f"Val FM loss: {val_loss:.6f} | Val relative FM loss {val_relative_loss:.6f}")

# Cleanup
if ddp_world_size > 1:
    dist.destroy_process_group()
