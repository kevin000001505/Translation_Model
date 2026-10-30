"""Fine-tune one model on a prepared dataset, scoring it before and after.

Runs the same way on the local ROCm GPU and inside a Modal container: reads
``<run_dir>/train.jsonl`` and ``<run_dir>/eval.jsonl`` and writes weights to ``out_dir``.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import torch

from .evaluate import compare, score
from .progress import heartbeat
from .prompts import encode, end_marker, plain_text_template, resolve_family
from .runconfig import TrainConfig

Log = Callable[[str], None]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def qlora_supported() -> tuple[bool, str]:
    """Whether 4-bit bitsandbytes works on this machine's GPU (ROCm support varies by card)."""
    if not torch.cuda.is_available():
        return False, "no GPU visible to PyTorch"
    try:
        import bitsandbytes as bnb

        layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").to("cuda")
        layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
        torch.cuda.synchronize()
        return True, f"bitsandbytes {bnb.__version__} 4-bit works on {torch.cuda.get_device_name(0)}"
    except Exception as exc:  # ImportError, missing kernels for this GPU, ...
        return False, f"{type(exc).__name__}: {exc}"


def load_model(cfg: TrainConfig, log: Log):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .gemma3 import fix_rope

    token = os.environ.get("HF_TOKEN") or None
    source = cfg.model_id
    if cfg.gguf_file:
        from .gguf_convert import ensure_converted

        source = str(ensure_converted(cfg.model_id, cfg.gguf_file, log))
    tokenizer = AutoTokenizer.from_pretrained(source, token=token, trust_remote_code=cfg.trust_remote_code)
    config = AutoConfig.from_pretrained(source, token=token, trust_remote_code=cfg.trust_remote_code)
    if fix_rope(config):
        log("Restored Gemma 3's linear RoPE scaling on the full-attention layers (lost in this copy's config).")
    kwargs = {"token": token, "trust_remote_code": cfg.trust_remote_code, "dtype": torch.bfloat16,
              "config": config}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    if cfg.method == "QLoRA":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    with heartbeat(log, f"loading {cfg.model_id} onto the {'GPU' if torch.cuda.is_available() else 'CPU'}"):
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if tokenizer.eos_token is None:
        # Some tokenizers (e.g. Seed-X) only declare EOS in the model config; without it the model
        # would never learn where a translation ends.
        eos_id = model.generation_config.eos_token_id or model.config.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        if eos_id is not None:
            tokenizer.eos_token = tokenizer.convert_ids_to_tokens(eos_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Loaded {cfg.model_id} ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, {cfg.method}).")
    return model, tokenizer, source


# Attention and MLP projections of the language model (the usual LoRA targets); the regex leaves out
# a vision tower, which multimodal checkpoints such as TranslateGemma load alongside it.
LORA_TARGETS = r"^(?!.*(vision|multi_modal_projector)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def add_adapters(model, cfg: TrainConfig, log: Log):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.method == "QLoRA":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = dict(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, task_type="CAUSAL_LM")
    try:
        model = get_peft_model(model, LoraConfig(target_modules=LORA_TARGETS, **lora))
    except ValueError:  # unusual module names: fall back to every linear layer
        model = get_peft_model(model, LoraConfig(target_modules="all-linear", **lora))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"LoRA r={cfg.lora_r}: {trainable / 1e6:.1f}M trainable parameters.")
    return model


class _Examples(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], tokenizer, family: str, cfg: TrainConfig, log: Log):
        self.items = []
        skipped = 0
        for r in rows:
            p, c = encode(tokenizer, family, cfg.src, cfg.tgt, r["src"], r["tgt"])
            if len(p) + len(c) > cfg.max_seq_len:
                skipped += 1
                continue
            self.items.append((p + c, [-100] * len(p) + c))  # loss on the translation only
        if skipped:
            log(f"Skipped {skipped} pairs longer than {cfg.max_seq_len} tokens.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _collate(pad_id: int):
    def collate(batch):
        n = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, : len(x)] = torch.tensor(x)
            labels[i, : len(y)] = torch.tensor(y)
            mask[i, : len(x)] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate


@torch.no_grad()
def translate(model, tokenizer, family: str, cfg: TrainConfig, sources: list[str], batch_size: int = 16,
              max_new_tokens: int = 160) -> list[str]:
    model.eval()
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    out: list[str] = []
    device = next(model.parameters()).device
    # Stop at the end of the assistant turn too (e.g. Gemma's <end_of_turn>), not only at EOS.
    stop_ids = {tokenizer.eos_token_id}
    if end_marker(tokenizer):
        stop_ids.add(tokenizer.convert_tokens_to_ids(end_marker(tokenizer)))
    stop_ids -= {None, tokenizer.unk_token_id}
    for i in range(0, len(sources), batch_size):
        prompts = [encode(tokenizer, family, cfg.src, cfg.tgt, s)[0] for s in sources[i : i + batch_size]]
        n = max(len(p) for p in prompts)
        ids = torch.tensor([[tokenizer.pad_token_id] * (n - len(p)) + p for p in prompts], device=device)
        mask = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device=device)
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=sorted(stop_ids) or None)
        for row in gen[:, n:]:
            text = tokenizer.decode(row, skip_special_tokens=True)
            out.append(text.split("</think>")[-1].strip().split("\n")[0].strip())
    tokenizer.padding_side = side
    return out


def evaluate_model(model, tokenizer, family: str, cfg: TrainConfig, rows: list[dict], log: Log, label: str) -> dict:
    if not rows:
        return {}
    t0 = time.time()
    hyps = translate(model, tokenizer, family, cfg, [r["src"] for r in rows])
    result = score(hyps, [r["tgt"] for r in rows], cfg.tgt)
    log(f"{label}: chrF {result['chrf']:.2f}, BLEU {result['bleu']:.2f} on {len(rows)} sentences "
        f"({time.time() - t0:.0f}s).")
    result["samples"] = [{"src": r["src"], "ref": r["tgt"], "hyp": h} for r, h in list(zip(rows, hyps))[:5]]
    result["hyps"] = hyps  # every translation, for the per-sentence validation report
    return result


def _check_adapter_moved(model, cfg: TrainConfig, log: Log) -> None:
    """LoRA's B matrices start at zero: if they are still zero, training changed nothing."""
    if cfg.method == "Full":
        return
    biggest = max((float(p.detach().abs().max()) for n, p in model.named_parameters() if "lora_B" in n),
                  default=0.0)
    log(f"Adapter check: largest LoRA B weight is {biggest:.2e}.")
    if biggest == 0.0:
        log("WARNING: the LoRA adapter weights are still zero, so training did not update them.")


def run_training(cfg: TrainConfig, run_dir: Path, out_dir: Path, log: Log = print) -> dict:
    from transformers import Trainer, TrainerCallback, TrainingArguments

    torch.manual_seed(cfg.seed)
    train_rows = _read_jsonl(run_dir / "train.jsonl")
    eval_rows = _read_jsonl(run_dir / "eval.jsonl")[: cfg.eval_sentences]
    log(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}; "
        f"{len(train_rows)} training pairs, {len(eval_rows)} evaluation sentences.")

    steps = 5 if cfg.gguf_export else 4
    log(f"Step 1/{steps}: preparing the base model.")
    model, tokenizer, source = load_model(cfg, log)
    family = resolve_family(cfg.family, tokenizer)

    log(f"Step 2/{steps}: scoring the base model on {len(eval_rows)} held-out sentences.")
    with heartbeat(log, "scoring the base model"):
        base_scores = evaluate_model(model, tokenizer, family, cfg, eval_rows, log, "Base model")

    if cfg.method in ("LoRA", "QLoRA"):
        model = add_adapters(model, cfg, log)
    model.config.use_cache = False
    data = _Examples(train_rows, tokenizer, family, cfg, log)
    if not len(data):
        raise ValueError("No training examples after tokenization.")
    eval_data = _Examples(eval_rows, tokenizer, family, cfg, log)
    curves: dict[str, list] = {"train": [], "validation": []}  # [step, loss] points for the plots

    class _LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                curves["train"].append([state.global_step, round(float(logs["loss"]), 4)])
                log(f"step {state.global_step}/{state.max_steps} epoch {logs.get('epoch', 0):.2f} "
                    f"loss {logs['loss']:.4f} lr {logs.get('learning_rate', 0):.2e}")
            if logs and "eval_loss" in logs:
                points = curves["validation"]
                if points and points[-1][0] == state.global_step:
                    return  # the final evaluation repeats the last scheduled one
                points.append([state.global_step, round(float(logs["eval_loss"]), 4)])
                if 0 < state.global_step < state.max_steps:  # before/after get their own log lines
                    log(f"step {state.global_step}/{state.max_steps} validation loss {logs['eval_loss']:.4f}")

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    steps_per_epoch = math.ceil(len(data) / (cfg.batch_size * cfg.grad_accum))
    total_steps = math.ceil(steps_per_epoch * cfg.epochs)
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total_steps)),  # an int works on transformers 4.x and 5.x
        logging_steps=max(1, min(10, steps_per_epoch // 5)),
        # Validation loss on the held-out titles about ten times per run, for the loss curves.
        eval_strategy="steps" if len(eval_data) else "no",
        eval_steps=max(1, total_steps // 10),
        per_device_eval_batch_size=cfg.batch_size,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        bf16=bf16,
        gradient_checkpointing=torch.cuda.is_available(),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if cfg.method == "QLoRA" else "adamw_torch",
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        disable_tqdm=True,  # the log callback reports progress; tqdm bars clutter the job log
    )
    trainer = Trainer(model=model, args=args, train_dataset=data, eval_dataset=eval_data if len(eval_data) else None,
                      data_collator=_collate(tokenizer.pad_token_id),
                      callbacks=[_LogCallback()])
    loss_before = None
    if len(eval_data):
        with heartbeat(log, "measuring the validation loss"):
            loss_before = float(trainer.evaluate()["eval_loss"])
        log(f"Validation loss before training: {loss_before:.4f}")
    log(f"Step 3/{steps}: training ({math.ceil(total_steps)} steps; progress is logged every few steps).")
    t0 = time.time()
    result = trainer.train()
    train_seconds = time.time() - t0
    log(f"Training finished in {train_seconds / 60:.1f} min, final loss {result.training_loss:.4f}.")
    loss_after = float(trainer.evaluate()["eval_loss"]) if len(eval_data) else None
    if loss_after is not None:
        log(f"Validation loss after training: {loss_after:.4f} (before: {loss_before:.4f})")
    _check_adapter_moved(model, cfg, log)

    model.config.use_cache = True
    log(f"Step 4/{steps}: scoring the fine-tuned model and saving it.")
    with heartbeat(log, "scoring the fine-tuned model"):
        tuned_scores = evaluate_model(model, tokenizer, family, cfg, eval_rows, log, "Fine-tuned model")

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / ("adapter" if cfg.method != "Full" else "model")
    with heartbeat(log, "saving the weights"):
        model.save_pretrained(weights_dir)
        tokenizer.save_pretrained(weights_dir)
    log(f"Saved weights to {weights_dir}")
    validation = {"loss_before": loss_before, "loss_after": loss_after}
    if eval_rows and base_scores and tuned_scores:
        details, summary = compare(eval_rows, base_scores.pop("hyps"), tuned_scores.pop("hyps"))
        validation.update(summary)
        with (out_dir / "validation.jsonl").open("w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        log(f"Validation: {summary['improved']} sentences better, {summary['worse']} worse, "
            f"{summary['sentences'] - summary['changed']} unchanged of {summary['sentences']} "
            "(details in validation.jsonl and on the Models tab).")
        if summary["changed"] == 0:
            log("WARNING: the fine-tuned model gave exactly the base model's translation for every validation "
                "sentence, so the fine-tune had no effect. Check the training and validation losses above.")
    metrics = {
        "base": base_scores,
        "tuned": tuned_scores,
        "train_loss": result.training_loss,
        "train_seconds": round(train_seconds),
        "train_examples": len(data),
        "prompt_family": family,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "validation": validation,
        "curves": curves,
    }
    if cfg.gguf_export:
        try:
            api_template = plain_text_template(tokenizer, family, cfg.src, cfg.tgt)
        except Exception as exc:  # the export then keeps the base model's own template
            log(f"Could not build a plain-text chat template ({exc}); keeping the base model's.")
            api_template = None
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        from .gguf_export import export_gguf

        try:
            base_gguf = None
            if cfg.gguf_file:  # merge into the original GGUF, keeping its tokenizer (already in the HF cache)
                from huggingface_hub import hf_hub_download

                token = os.environ.get("HF_TOKEN") or None
                base_gguf = Path(hf_hub_download(cfg.model_id, cfg.gguf_file, token=token))
            log(f"Step 5/{steps}: exporting a {cfg.gguf_export} GGUF.")
            with heartbeat(log, "exporting the GGUF"):
                path = export_gguf(cfg.gguf_export, source, weights_dir, cfg.method != "Full", out_dir,
                                   run_dir / "gguf-work", cfg.trust_remote_code, log, base_gguf=base_gguf,
                                   chat_template=api_template)
            metrics["gguf"] = {"type": cfg.gguf_export, "file": str(path.relative_to(out_dir)),
                               "plain_text_template": api_template is not None}
        except Exception as exc:  # the fine-tune itself is fine; report the export problem and keep it
            log(f"GGUF export failed: {exc}")
            metrics["gguf"] = {"type": cfg.gguf_export, "error": str(exc)}
    return metrics
