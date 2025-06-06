import os
import sys
import json
import random
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# ---- SYNTHETIC DATA GENERATION ----
def make_fake_dataset(num_samples=8, max_length=32):
    # Generate some trivial text data
    texts = [
        f"This is fake sample {i}. Some more fake text here. Value={random.randint(1, 100)}."
        for i in range(num_samples)
    ]
    return texts

@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    output_dir: str = "./gpt2_lora_finetune"
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    max_length: int = 32
    seed: int = 42

def prepare_dataset(texts, tokenizer, max_length):
    encodings = tokenizer(
        texts, truncation=True, padding="max_length", max_length=max_length
    )
    return Dataset.from_dict(encodings)

def train(cfg: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    texts = make_fake_dataset(num_samples=8, max_length=cfg.max_length)
    train_dataset = prepare_dataset(texts, tokenizer, cfg.max_length)

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["c_attn"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        save_steps=5,
        save_total_limit=1,
        fp16=False,
        report_to=[],
        seed=cfg.seed,
        logging_steps=2,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Model saved to {cfg.output_dir}")

    return cfg.output_dir

def test_vllm(model_dir, test_prompt="This is a test input."):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vllm not installed; skipping vllm test.")
        return

    print("Testing vLLM inference...")
    llm = LLM(model=model_dir)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=32)
    outputs = llm.generate([test_prompt], sampling_params)
    for output in outputs:
        print("vLLM output:", output.outputs[0].text)

def main():
    cfg = TrainingConfig()
    model_dir = train(cfg)
    test_vllm(model_dir)

if __name__ == "__main__":
    main()
