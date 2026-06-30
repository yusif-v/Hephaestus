"""Training engine — SFT with learned optimizations."""

import json
import time
import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset as HFDataset


def train_model(model, tokenizer, train_data, config):
    """Train model with SFT. Returns training result and metrics."""

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
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=hf_dataset,
        args=args,
    )

    print(f"Starting training: {config.training.max_steps} steps, "
          f"batch={config.training.batch_size}, lr={config.training.learning_rate}")

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    training_metrics = {
        "train_time_minutes": round(elapsed / 60, 1),
        "final_loss": round(result.training_loss, 4),
        "steps_completed": result.global_step,
    }

    print(f"Training complete: {training_metrics['train_time_minutes']} min, "
          f"loss={training_metrics['final_loss']}")

    return result, training_metrics
