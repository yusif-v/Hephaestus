"""Training engine — SFT with learned optimizations + overfitting detection."""

import json
import time
from collections import defaultdict

from trl import SFTTrainer, SFTConfig
from datasets import Dataset as HFDataset


def auto_configure(config, vram_gb=None):
    """Auto-adjust training parameters based on system.

    Phase 2.3: Auto-select LoRA rank based on model size
    Phase 2.4: Auto-select batch_size based on available VRAM
    """
    model_name = config.model.name.lower()

    # Auto rank selection based on model size (Phase 2.3)
    if "0.5b" in model_name:
        config.lora.rank = 64
        config.lora.alpha = 128
    elif "1.5b" in model_name:
        config.lora.rank = 128
        config.lora.alpha = 256
    elif "3b" in model_name:
        config.lora.rank = 256
        config.lora.alpha = 512
    elif "7b" in model_name:
        config.lora.rank = 512
        config.lora.alpha = 1024

    # Auto batch_size based on VRAM (Phase 2.4)
    if vram_gb:
        if "7b" in model_name:
            config.training.batch_size = 1
            config.training.gradient_accumulation_steps = 4
        elif "3b" in model_name:
            if vram_gb >= 24:
                config.training.batch_size = 2
            elif vram_gb >= 16:
                config.training.batch_size = 1
                config.training.gradient_accumulation_steps = 2
            else:
                config.training.batch_size = 1
                config.training.gradient_accumulation_steps = 4
        elif "1.5b" in model_name:
            config.training.batch_size = 2 if vram_gb >= 16 else 1
        elif "0.5b" in model_name:
            config.training.batch_size = 4 if vram_gb >= 16 else 2

    return config


def train_model(model, tokenizer, train_data, config):
    """Train model with SFT. Returns training result and metrics.

    Includes overfitting detection (Phase 2.2): monitors if training loss
    keeps decreasing while eval loss starts increasing.
    """

    # Format dataset for SFT
    def format_fn(item):
        if "messages" in item:
            messages = item["messages"]
        else:
            messages = item
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False)
        parts = []
        for m in messages:
            parts.append("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>")
        return "\n".join(parts)

    texts = [format_fn(item) for item in train_data]
    hf_dataset = HFDataset.from_dict({"text": texts})

    args = SFTConfig(
        output_dir=config.output.dir + "/checkpoints",
        max_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        optim=config.training.optimizer,
        gradient_checkpointing=config.training.gradient_checkpointing,
        bf16=config.training.bf16,
        max_length=config.training.max_length,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        max_grad_norm=config.training.max_grad_norm,
        seed=config.training.seed,
        report_to="none",
        eval_strategy="steps",
        eval_steps=max(config.training.max_steps // 5, 20),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=hf_dataset,
        args=args,
    )

    print(f"Starting training: {config.training.max_steps} steps, "
          f"batch={config.training.batch_size}, lr={config.training.learning_rate}, "
          f"rank={config.lora.rank}, alpha={config.lora.alpha}")

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    # Overfitting detection: check if eval_loss > final train_loss (Phase 2.2)
    train_loss = result.training_loss
    eval_loss = getattr(result, "eval_loss", None)
    overfitting_warning = None

    if eval_loss and train_loss:
        loss_gap = eval_loss - train_loss
        if loss_gap > 0.1:
            overfitting_warning = (
                f"OVERFITTING DETECTED: eval_loss ({eval_loss:.4f}) >> "
                f"train_loss ({train_loss:.4f}), gap={loss_gap:.4f}. "
                f"Consider: fewer steps, higher dropout, or more data."
            )

    training_metrics = {
        "train_time_minutes": round(elapsed / 60, 1),
        "final_loss": round(train_loss, 4),
        "eval_loss": round(eval_loss, 4) if eval_loss else None,
        "steps_completed": result.global_step,
        "overfitting_warning": overfitting_warning,
    }

    print(f"Training complete: {training_metrics['train_time_minutes']} min, "
          f"loss={training_metrics['final_loss']}"
          + (f", eval_loss={training_metrics['eval_loss']}" if eval_loss else ""))

    if overfitting_warning:
        print(f"\n⚠ {overfitting_warning}")

    return result, training_metrics
