"""QLoRA fine-tune, plain Hugging Face stack (no Unsloth). Single- or multi-GPU.

Recipe: r=32 alpha=64 all-linear, lr 2e-4 cosine, <=3 epochs, early stop at
best dev loss. The default model is the full-precision public Ministral 3 14B,
quantized to 4-bit on the fly (no HF token needed).

Multi-GPU (DDP): when launched via torchrun (launcher --multi-gpu), every GPU
on the box trains in parallel on different batches. Gradient accumulation is
divided by the number of GPUs so the effective batch stays 16 -- identical
training math, ~Nx faster wall clock. Single-GPU behavior unchanged otherwise.

Record keeping: every checkpoint kept under /opt/ml/checkpoints (live-synced
to S3), metrics.jsonl logs step/epoch/learning_rate/loss, run_config.json
snapshots hyperparameters, and interrupted jobs resume from last checkpoint.
"""
import argparse
import json
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Ministral-3-14B-Instruct-2512")
    p.add_argument("--data_dir", default=os.environ.get("SM_CHANNEL_TRAIN", "../../data/train"))
    p.add_argument("--train_file", default="rehearsal_train.jsonl")
    p.add_argument("--output_dir", default=os.environ.get("SM_MODEL_DIR", "out"))
    p.add_argument("--gpu", default="0", help="CUDA device index (single-GPU mode only)")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8,
                   help="divided by GPU count in multi-GPU mode (effective batch constant)")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--eval_holdout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    # torchrun sets LOCAL_RANK/WORLD_SIZE; absent means single-GPU mode.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = local_rank <= 0
    if local_rank == -1:
        # Single-GPU mode: pin to the requested card before CUDA init.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    import inspect

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, EarlyStoppingCallback,
                              TrainerCallback)
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    ckpt_dir = ("/opt/ml/checkpoints" if os.path.isdir("/opt/ml/checkpoints")
                else os.path.join(args.output_dir, "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        record = dict(vars(args), world_size=world_size)
        with open(os.path.join(ckpt_dir, "run_config.json"), "w") as f:
            json.dump(record, f, indent=2)

    metrics_path = os.path.join(ckpt_dir, "metrics.jsonl")

    class JsonlLogger(TrainerCallback):
        def on_log(self, targs, state, control, logs=None, **kwargs):
            if logs and state.is_world_process_zero:
                rec = {"step": state.global_step, "epoch": state.epoch, **logs}
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

    tok = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None if "4bit" in args.model.lower() else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    load_kwargs = dict(device_map={"": max(local_rank, 0)},
                       dtype=torch.bfloat16, quantization_config=quant)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except ValueError:
        # Ministral 3 is a vision+text model; its config registers under the
        # image-text auto-class even when used purely for text.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        # Regex: attach adapters ONLY inside the language model. The vision
        # tower never runs on text-only batches; adapters there would get no
        # gradients and crash DDP's reducer.
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")

    train_path = os.path.join(args.data_dir, args.train_file)
    ds = load_dataset("json", data_files=train_path, split="train")
    ds = ds.map(
        lambda row: {"text": tok.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False)},
        remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=args.eval_holdout, seed=args.seed)

    def accepted(cls, kwargs):
        params = inspect.signature(cls.__init__).parameters
        return {k: v for k, v in kwargs.items() if k in params}

    grad_accum = max(1, args.grad_accum // world_size)

    cfg = SFTConfig(**accepted(SFTConfig, dict(
        output_dir=ckpt_dir,
        dataset_text_field="text",
        max_length=args.max_seq_len,
        max_seq_length=args.max_seq_len,
        learning_rate=args.lr, lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        eval_strategy="steps", eval_steps=25,
        save_strategy="steps", save_steps=25,  # keep every checkpoint
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=10, seed=args.seed, report_to="none")))

    trainer_kwargs = dict(
        model=model, args=cfg, peft_config=peft_config,
        train_dataset=split["train"], eval_dataset=split["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), JsonlLogger()])
    tokenizer_param = ("processing_class"
                       if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
                       else "tokenizer")
    trainer_kwargs[tokenizer_param] = tok
    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=get_last_checkpoint(ckpt_dir))

    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        trainer.save_model(os.path.join(args.output_dir, "adapter"))
        tok.save_pretrained(os.path.join(args.output_dir, "adapter"))
        trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))
        for fname in ("metrics.jsonl", "run_config.json"):
            src = os.path.join(ckpt_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.output_dir, fname))


if __name__ == "__main__":
    main()
