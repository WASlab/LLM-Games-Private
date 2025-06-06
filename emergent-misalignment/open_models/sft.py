import os, json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, DataCollatorForSeq2Seq
)
from trl import SFTTrainer, train_on_responses_only
from peft import get_peft_model, LoraConfig

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def prepare_dataset(cfg):
    rows = load_jsonl(cfg["training_file"])
    train_ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    if cfg.get("test_file"):
        rows_test = load_jsonl(cfg["test_file"])
        test_ds = Dataset.from_list([{"messages": r["messages"]} for r in rows_test])
    else:
        split = train_ds.train_test_split(test_size=0.1, seed=cfg["seed"])
        train_ds, test_ds = split["train"], split["test"]
    return train_ds, test_ds

def get_peft(model, cfg):
    lora_cfg = LoraConfig(
        r=cfg["r"], lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"], lora_dropout=cfg["lora_dropout"],
        bias=cfg["lora_bias"], task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_cfg)

def main(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if cfg["is_peft"]:
        model = get_peft(model, cfg)

    train_ds, test_ds = prepare_dataset(cfg)

    # HARD CODE for Gemma 3 chat template:
    instruction_part = "<start_of_turn>user\n"
    response_part = "<start_of_turn>model\n"

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            num_train_epochs=cfg["epochs"],
            optim=cfg["optim"],
            weight_decay=cfg["weight_decay"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            logging_steps=cfg["logging_steps"],
            save_steps=cfg["save_steps"],
            seed=cfg["seed"],
            bf16=torch.cuda.is_available(),
            report_to=None,
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        dataset_text_field="messages",
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        max_seq_length=cfg["max_seq_length"],
        packing=False,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part
    )
    trainer.train()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
