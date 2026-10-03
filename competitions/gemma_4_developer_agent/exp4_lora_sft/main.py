"""
Exp 4 -- First LoRA SFT pass on gemma-4-e4b-it against the synthetic
trajectory dataset built from tasks.jsonl's 129 public dev tasks (see
competitions/gemma_4_developer_agent/data/sft_trajectories.jsonl and the
builder script that made it; the dataset itself is mounted here as the
Kaggle Dataset `daltongabrielomondi/gemma4-sft-trajectories`).

WHY: exp1-3 (zero-shot, no training) established that gemma-4-e4b-it does
NOT reliably speak the documented ```tool_call\\n{"tool":...,"args":{...}}\\n```
convention on its own -- it drifts to a bare `call: name(args)` pseudo-Python
form (see THINKING_AND_DECISIONS.md, exp1's real transcripts). Nobody has
real successful agent trajectories to imitate (the real swegemma harness
isn't locally runnable), so this SFT pass uses TEMPLATED trajectories: a
fixed policy that reads exactly the region of each file a hunk touches,
edits it via the documented tool-call syntax, and submits -- built
directly from the ground-truth reference patches. This is behavior
cloning against synthetic rollouts, not real agent demonstrations --
flagged as a real methodology risk in THINKING_AND_DECISIONS.md (it can
teach a narrow "read once, edit once, submit" pattern rather than the
exploratory reasoning needed when the fix location ISN'T already known,
which is the normal eval-time case). Iteration 1: get tool-call syntax
fluent and expose real bug-fix diffs; not a claim of a finished recipe.

Runs entirely on Kaggle (GPU kernel, T4/P100). Saves the trained adapter
to /kaggle/working/adapter/ in standard PEFT format (adapter_config.json +
adapter_model.safetensors) -- ready to drop into a submission.zip's
adapters/main_lora/.
"""
import gc
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Reduce allocator fragmentation risk on a tight-memory single GPU (T4/P100,
# ~14.5GB usable) -- must be set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))
WARMUP_RATIO = 0.03
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = LORA_R * 2
LORA_DROPOUT = 0.05
VAL_FRACTION = 0.09  # ~11/121 held out for a val-loss signal, not real eval
SEED = 42

WORK_ROOT = Path("/kaggle/working")
RESULTS_DIR = WORK_ROOT / "results"
ADAPTER_DIR = WORK_ROOT / "adapter"
for d in (RESULTS_DIR, ADAPTER_DIR):
    d.mkdir(parents=True, exist_ok=True)

random.seed(SEED)


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Locate SFT dataset + model on this Kaggle kernel
# ---------------------------------------------------------------------------
def find_file(candidates_globs):
    for pattern in candidates_globs:
        for hit in sorted(glob.glob(pattern, recursive=True)):
            if Path(hit).is_file():
                return Path(hit)
    return None


def find_dir(candidates_globs):
    for pattern in candidates_globs:
        for hit in sorted(glob.glob(pattern, recursive=True)):
            p = Path(hit)
            if p.is_dir():
                return p
    return None


SFT_DATA_PATH = find_file([
    "/kaggle/input/gemma4-sft-trajectories/sft_trajectories.jsonl",
    "/kaggle/input/**/sft_trajectories.jsonl",
])
MODEL_DIR = find_dir([
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/*",
    "/kaggle/input/models/google/gemma-4/*/gemma-4-e4b-it/*",
    "/kaggle/input/gemma-4/transformers/gemma-4-e4b-it/*",
    "/kaggle/input/gemma-4/*/gemma-4-e4b-it/*",
    "/kaggle/input/**/gemma-4-e4b-it/*",
])

log(f"SFT_DATA_PATH = {SFT_DATA_PATH}")
log(f"MODEL_DIR = {MODEL_DIR}")
if SFT_DATA_PATH is None or MODEL_DIR is None:
    log("FATAL: could not resolve dataset or model directory. Listing /kaggle/input for diagnosis:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
        if depth > 6:
            dirs[:] = []
            continue
        log(f"  {root}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load synthetic trajectories
# ---------------------------------------------------------------------------
examples_raw = []
with open(SFT_DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            examples_raw.append(json.loads(line))
log(f"Loaded {len(examples_raw)} synthetic trajectories")

random.shuffle(examples_raw)
n_val = max(1, int(len(examples_raw) * VAL_FRACTION))
val_raw = examples_raw[:n_val]
train_raw = examples_raw[n_val:]
log(f"Split: {len(train_raw)} train / {len(val_raw)} val")

# ---------------------------------------------------------------------------
# Install / import deps (same pattern as exp1-3: force-upgrade since Gemma 4
# is brand new and the Kaggle base image may predate architecture support)
# ---------------------------------------------------------------------------
for pkg in ("transformers", "accelerate", "bitsandbytes", "peft"):
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", pkg],
                            capture_output=True, text=True, timeout=240)
        log(f"pip install -U {pkg}: exit={r.returncode}" + (f" stderr_tail={r.stderr[-300:]}" if r.returncode else ""))
    except Exception as e:
        log(f"pip install -U {pkg} failed: {e}")

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
log(f"transformers version: {__import__('transformers').__version__}")
log(f"peft version: {__import__('peft').__version__}")

try:
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    tokenizer = getattr(processor, "tokenizer", processor)
except Exception as e:
    log(f"AutoProcessor failed ({e}), falling back to AutoTokenizer")
    processor = None
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# gemma-4-e4b-it turns out to be a real ~8B-parameter multi-modal
# checkpoint (audio tower + presumably vision + the text decoder we
# actually want), not a literal 4B model -- confirmed via a real run's
# `peft.print_trainable_parameters()` output ("all params: 7,981,684,000").
# In 4-bit that alone resident on ONE ~14.5GB Kaggle T4/P100 left only
# ~1GB of headroom for any forward/backward activations (confirmed: OOM'd
# on the very FIRST training example with "13.36 GiB already in use, 1.2
# GiB free" before any real activation memory was even requested) --
# device_map="auto" packed the entire model onto a single GPU without
# sharding, even though this is a Kaggle T4x2 kernel (2 physical GPUs).
# Force a memory cap per GPU so accelerate's automatic placement has to
# shard the model across both, roughly doubling the headroom available
# for LoRA activations instead of leaving a second GPU completely idle.
_n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
_max_memory = None
if _n_gpus > 1:
    # A 9GiB/device cap (attempt 1) was too tight -- some atomic leaf
    # module didn't fit within budget on either device and accelerate
    # fell back to disallowed CPU offload. A 13GiB/device cap (attempt 2)
    # was too LOOSE the other way -- the real 4-bit base model only turned
    # out to be ~9.3GB total, comfortably under 13GiB, so accelerate just
    # packed the whole thing onto GPU 1 again (confirmed via a real run's
    # explicit per-device memory log: GPU 0 stayed at 0.00GB the entire
    # time) -- i.e. neither cap actually forced a split; 13GiB just also
    # happened not to trigger CPU-offload. That run then OOM'd on every
    # single example across all 3 epochs (500s, zero real optimizer steps)
    # because ~13.3GB was resident before any activation memory was even
    # requested, leaving only ~1.2GB headroom on the one GPU actually used
    # -- identical failure mode to not capping at all, just reached a
    # different way. Real fix: cap each device to LESS than half the
    # model's actual ~9.3GB footprint, so accelerate has no choice but to
    # split it across both GPUs.
    # 5GiB was ALSO too tight -- same CPU-offload ValueError as the 9GiB
    # attempt. Bounding from both failures: some atomic leaf module (must
    # be the embedding table / tied lm_head -- the only plausible
    # multi-GB un-splittable single tensor, and consistent with v1's
    # fp32-upcast OOM needing ~10.5GB, i.e. ~5.25GB in bf16) doesn't fit
    # in a 5GiB budget but does fit in a 9GiB one. 6GiB should clear that
    # module on whichever device gets it, while still being under half
    # the ~9.3GB total -- forcing the remaining ~4GB of transformer
    # layers onto the OTHER device.
    _max_memory = {i: "6GiB" for i in range(_n_gpus)}
    log(f"{_n_gpus} GPUs visible -- capping each to 6GiB via max_memory to FORCE a real split "
        f"(5GiB was too tight -> CPU offload error again; 13GiB was too loose -> single-GPU "
        f"packing, 0 real steps in 500s; 6GiB should clear the ~5.25GB embedding table on "
        f"whichever device gets it while still forcing the rest onto the other device).")

log("Loading base model (4-bit NF4)...")
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=_max_memory,
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
log("Base model loaded.")
if torch.cuda.is_available():
    for _dev in range(_n_gpus):
        log(f"GPU {_dev} mem after base load: allocated={torch.cuda.memory_allocated(_dev)/1e9:.2f}GB "
            f"reserved={torch.cuda.memory_reserved(_dev)/1e9:.2f}GB")
    try:
        log(f"hf_device_map: {model.hf_device_map}")
    except Exception:
        pass

# NOTE: deliberately NOT calling peft's prepare_model_for_kbit_training()
# here. Its default behavior upcasts EVERY non-4bit-quantized parameter
# (anything bitsandbytes didn't turn into Params4bit -- notably the
# embedding table / tied lm_head, which bnb never quantizes) to float32.
# On this model that upcast alone tried to allocate 10.5GB and OOM'd on a
# ~14.5GB single T4/P100 that already had ~11.4GB resident just from the
# 4-bit base weights (exp4's first push failed exactly this way -- see
# THINKING_AND_DECISIONS.md). LoRA training doesn't need the frozen base
# model's non-quantized params in fp32 -- they stay in bf16 (matching
# bnb_4bit_compute_dtype above) and are never updated anyway (requires_grad
# is set False on all of them). Only the small LoRA adapter params (added
# by get_peft_model below) need real gradients / an fp32-ish dtype, and
# peft creates those correctly on its own.
for param in model.parameters():
    param.requires_grad = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Discover the real target_modules strings rather than assuming the
# standard Llama-style leaf names ("q_proj" etc.) are directly wrappable.
# gemma-4-e4b-it's "elastic" architecture wraps attention/MLP projections
# in a custom `Gemma4ClippableLinear` module (presumably the mechanism
# behind its dynamic/elastic effective size) whose real `Linear4bit` lives
# one level down as a `.linear` child -- PEFT's LoRA injector doesn't
# recognize `Gemma4ClippableLinear` itself as a supported layer type and
# raises ValueError on it (confirmed via a real failed run).
#
# A first fix attempt used leaf-suffix target strings (e.g. "q_proj" and
# "q_proj.linear" together), which ALSO failed: the architecture is
# heterogeneous -- discovery found BOTH plain Linear4bit "q_proj" leaves
# AND Gemma4ClippableLinear-wrapped ones with a ".linear" child, evidently
# varying by layer (plausibly a QAT-style precision exception for
# boundary layers). PEFT's target-module matching is suffix-based and
# layer-agnostic, so including bare "q_proj" as a target ALSO matched the
# *wrapper* module path on the layers that use the wrapped form -- the
# exact same crash, just reached via the other target string.
#
# Real fix: target exact, FULL dotted module paths (not shared leaf
# suffixes) for every discovered real Linear/Linear4bit leaf, one per
# layer. A full path can only exact/endswith-match its own unique module,
# so there is no cross-layer ambiguity regardless of which layers are
# wrapped and which aren't.
_CANDIDATE_PROJ_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
try:
    import bitsandbytes as bnb
    _LINEAR_TYPES = (torch.nn.Linear, bnb.nn.Linear4bit)
except Exception:
    _LINEAR_TYPES = (torch.nn.Linear,)

_discovered_targets = set()
_module_by_name = dict(model.named_modules())
for name, module in _module_by_name.items():
    leaf = name.rsplit(".", 1)[-1] if "." in name else name
    if leaf not in _CANDIDATE_PROJ_NAMES:
        continue
    if isinstance(module, _LINEAR_TYPES):
        _discovered_targets.add(name)  # full path, already a real Linear
        continue
    # Not directly a Linear -- look one (or more) levels down for the real
    # Linear/Linear4bit child and target ITS full path.
    for subname, submodule in module.named_modules():
        if subname and isinstance(submodule, _LINEAR_TYPES):
            _discovered_targets.add(f"{name}.{subname}")
            break

if not _discovered_targets:
    log("WARNING: target-module discovery found nothing wrappable; falling back to plain leaf names "
        "(this will likely fail the same way the first attempt did).")
    _discovered_targets = set(_CANDIDATE_PROJ_NAMES)

# gemma-4-e4b-it is multi-modal -- discovery over the whole module tree
# picked up q/k/v projections inside `audio_tower` (and likely a vision
# tower too) alongside the actual text decoder (confirmed via a real run:
# 406 targets, sample paths under `model.audio_tower.layers...`). None of
# that is relevant to a text-only SWE-agent LoRA and it was eating GPU
# memory (and gradient-hook overhead) for modalities our task never uses
# -- exclude any path through a non-text tower.
_NON_TEXT_MARKERS = ("audio_tower", "vision_tower", "image_tower", "vision_model", "audio_model", "vision_encoder")
_excluded = {t for t in _discovered_targets if any(m in t for m in _NON_TEXT_MARKERS)}
if _excluded:
    log(f"Excluding {len(_excluded)} non-text-tower targets (sample: {sorted(_excluded)[:3]})")
    _discovered_targets -= _excluded

target_modules = sorted(_discovered_targets)
log(f"Target-module discovery: {len(target_modules)} full-path targets found "
    f"(sample: {target_modules[:4]})")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
trainable_params = [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable_params)
log(f"Trainable (LoRA) params: {n_trainable:,}")


# ---------------------------------------------------------------------------
# Render each trajectory into (input_ids, labels) with loss masked to
# assistant-turn tokens only, via incremental chat-template tokenization
# (standard technique for multi-turn SFT without per-token offset mapping:
# render the cumulative prefix through each message, diff token-length
# against the previous prefix to isolate that turn's tokens). Uses the
# EXACT same folding fallback as exp1-3's harness (fold system -> first
# user turn if the chat template rejects a bare system role) so training
# and inference speak the same rendered text.
# ---------------------------------------------------------------------------
def render_messages(messages):
    kwargs = {"add_generation_prompt": False, "tokenize": False}
    target = processor if processor is not None else tokenizer
    if processor is not None:
        kwargs["enable_thinking"] = False
    try:
        return target.apply_chat_template(messages, **kwargs)
    except Exception:
        folded = []
        sys_text = None
        for m in messages:
            if m["role"] == "system":
                sys_text = m["content"]
                continue
            if sys_text and m["role"] == "user" and not folded:
                folded.append({"role": "user", "content": f"{sys_text}\n\n{m['content']}"})
                sys_text = None
            else:
                folded.append(m)
        if not folded:
            return ""
        return target.apply_chat_template(folded, **kwargs)


def encode_text(text: str):
    if not text:
        return []
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_example(ex):
    messages = ex["messages"]
    loss_mask = ex["loss_mask"]
    input_ids = []
    labels = []
    prev_ids = []
    for i in range(len(messages)):
        try:
            prefix_text = render_messages(messages[:i + 1])
            ids = encode_text(prefix_text)
        except Exception:
            ids = prev_ids  # template choked on this partial slice (e.g. system-only prefix) -- no new tokens yet
        if len(ids) < len(prev_ids):
            # Shouldn't happen (prefix should only grow), but never let a
            # tokenizer quirk produce a negative-length segment.
            ids = prev_ids
        new_segment = ids[len(prev_ids):]
        is_assistant = loss_mask[i]
        input_ids.extend(new_segment)
        labels.extend(new_segment if is_assistant else [-100] * len(new_segment))
        prev_ids = ids
    return input_ids, labels


log("Tokenizing + masking trajectories...")
train_examples = []
n_skipped_long = 0
for ex in train_raw:
    ids, labels = build_example(ex)
    if len(ids) == 0 or not any(l != -100 for l in labels):
        continue
    if len(ids) > MAX_SEQ_LEN:
        n_skipped_long += 1
        continue
    train_examples.append({"instance_id": ex["instance_id"], "input_ids": ids, "labels": labels})

val_examples = []
for ex in val_raw:
    ids, labels = build_example(ex)
    if len(ids) == 0 or not any(l != -100 for l in labels) or len(ids) > MAX_SEQ_LEN:
        continue
    val_examples.append({"instance_id": ex["instance_id"], "input_ids": ids, "labels": labels})

log(f"Usable train examples: {len(train_examples)} (skipped {n_skipped_long} over MAX_SEQ_LEN={MAX_SEQ_LEN})")
log(f"Usable val examples: {len(val_examples)}")
seq_lens = [len(e["input_ids"]) for e in train_examples]
if seq_lens:
    log(f"Train seq len: min={min(seq_lens)} median={sorted(seq_lens)[len(seq_lens)//2]} max={max(seq_lens)}")

if not train_examples:
    log("FATAL: no usable training examples after tokenization/masking.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Manual training loop (batch size 1, grad accumulation -- dataset is only
# ~110 examples so a full Trainer/collator setup is unnecessary overhead
# for this first iteration; batch size 1 also sidesteps any padding/mask
# bugs since each example is fed to the model exactly as tokenized).
# ---------------------------------------------------------------------------
device = next(model.parameters()).device
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

steps_per_epoch = math.ceil(len(train_examples) / GRAD_ACCUM)
total_optimizer_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = max(1, int(total_optimizer_steps * WARMUP_RATIO))


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_optimizer_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def forward_loss(example):
    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([example["labels"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return out.loss


@torch.no_grad()
def eval_val_loss():
    if not val_examples:
        return None
    model.eval()
    losses = []
    for ex in val_examples:
        try:
            loss = forward_loss(ex)
            if torch.isfinite(loss):
                losses.append(loss.item())
        except Exception as e:
            log(f"  val example {ex['instance_id']} forward failed: {e}")
    model.train()
    return sum(losses) / len(losses) if losses else None


log(f"Training: {len(train_examples)} examples x {NUM_EPOCHS} epochs, "
    f"grad_accum={GRAD_ACCUM}, lr={LEARNING_RATE}, total_optimizer_steps={total_optimizer_steps}")

loss_log = []
model.train()
global_step = 0
t0 = time.time()
try:
    for epoch in range(NUM_EPOCHS):
        order = list(range(len(train_examples)))
        random.shuffle(order)
        optimizer.zero_grad()
        running_loss = 0.0
        running_n = 0
        for i, idx in enumerate(order):
            ex = train_examples[idx]
            # Forward AND backward both live inside this try -- an earlier
            # version only wrapped forward_loss(), so an OOM raised during
            # .backward() (which happened in practice: most of the first
            # real training run's examples OOM'd on the backward pass, not
            # the forward pass) escaped uncaught, hit the OUTER try around
            # the whole epoch loop, and aborted ALL remaining training
            # silently -- the run then "completed" and saved an adapter
            # that had taken ~0 real optimizer steps (see
            # THINKING_AND_DECISIONS.md). Catching OOM at this granularity
            # instead means one oversized example gets skipped, not the
            # entire rest of training.
            try:
                loss = forward_loss(ex)
                (loss / GRAD_ACCUM).backward()
                loss_value = loss.item()
            except torch.OutOfMemoryError as e:
                log(f"  OOM on {ex['instance_id']} (len={len(ex['input_ids'])}), skipping: {e}")
                optimizer.zero_grad()
                loss = None
                gc.collect()
                torch.cuda.empty_cache()
                continue
            finally:
                # Unconditional cleanup every example (not just on OOM) --
                # a real run showed "memory in use" creeping up across
                # consecutive OOM'd examples even with empty_cache() called
                # on the failure path (13.36 -> 14.55 GiB over ~12
                # examples), eventually corrupting the CUDA context
                # ("illegal memory access") badly enough that even the
                # final save_pretrained() call failed. Dropping the local
                # reference and forcing gc + empty_cache after EVERY
                # example (success or failure) is cheap insurance against
                # that creep.
                loss = None
                gc.collect()
                torch.cuda.empty_cache()
            running_loss += loss_value
            running_n += 1

            if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                global_step += 1
                avg = running_loss / max(1, running_n)
                elapsed = time.time() - t0
                mem = (f" gpu_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB"
                       if torch.cuda.is_available() else "")
                log(f"epoch {epoch+1}/{NUM_EPOCHS} step {global_step}/{total_optimizer_steps} "
                    f"loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e} elapsed={elapsed:.0f}s{mem}")
                loss_log.append({"epoch": epoch + 1, "global_step": global_step,
                                  "train_loss": avg, "lr": scheduler.get_last_lr()[0],
                                  "elapsed_s": round(elapsed, 1)})
                running_loss = 0.0
                running_n = 0

        val_loss = eval_val_loss()
        log(f"=== epoch {epoch+1} done. val_loss={val_loss} ===")
        loss_log.append({"epoch": epoch + 1, "epoch_end": True, "val_loss": val_loss})
except Exception as e:
    log(f"Training loop raised: {e}\n{traceback.format_exc()}")

total_time = time.time() - t0
log(f"Training finished in {total_time:.0f}s")

n_real_optimizer_steps = sum(1 for entry in loss_log if "global_step" in entry)
training_effectively_empty = n_real_optimizer_steps == 0
if training_effectively_empty:
    # Don't let a silent early abort masquerade as a trained adapter --
    # this happened for real (see THINKING_AND_DECISIONS.md): an
    # uncaught OOM during backward() blew past the whole epoch loop
    # before a single optimizer.step() ran, the script still reached
    # save_pretrained() normally, and the kernel reported COMPLETE with
    # zero indication anything was wrong short of reading the raw log.
    log("WARNING *** loss_log has ZERO completed optimizer steps -- training did not "
        "meaningfully update the adapter. The saved adapter below is effectively "
        "still at LoRA initialization, not a trained result. ***")

# ---------------------------------------------------------------------------
# Save adapter in standard PEFT format
# ---------------------------------------------------------------------------
model.save_pretrained(str(ADAPTER_DIR))
log(f"Adapter saved to {ADAPTER_DIR}")
adapter_files = sorted(p.name for p in ADAPTER_DIR.glob("*"))
log(f"Adapter dir contents: {adapter_files}")

summary = {
    "note": "Exp 4 -- first LoRA SFT pass on synthetic templated trajectories (see module docstring for the real-vs-templated-trajectory caveat).",
    "model": "gemma-4-e4b-it (4-bit NF4 QLoRA)",
    "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                     "n_target_modules": len(target_modules),
                     "target_modules_sample": target_modules[:8]},
    "n_train_examples": len(train_examples),
    "n_val_examples": len(val_examples),
    "n_skipped_over_max_seq_len": n_skipped_long,
    "max_seq_len": MAX_SEQ_LEN,
    "num_epochs": NUM_EPOCHS,
    "grad_accum": GRAD_ACCUM,
    "learning_rate": LEARNING_RATE,
    "total_optimizer_steps": total_optimizer_steps,
    "n_real_optimizer_steps_completed": n_real_optimizer_steps,
    "training_effectively_empty": training_effectively_empty,
    "trainable_params": n_trainable,
    "total_train_time_s": round(total_time, 1),
    "loss_log": loss_log,
    "adapter_files": adapter_files,
}
(RESULTS_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
log("DONE.")
