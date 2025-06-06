import os, sys, json, backoff, torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model  # still needed for LoRA
from validate import TrainingConfig
from utils import load_jsonl                               # keep your helpers


def load_model_and_tokenizer(model_name: str, load_in_4bit: bool = False):
    """
    Pure-Transformers loader. 4-bit / 8-bit handled via bitsandbytes if desired.
    """
    quant_args = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_args["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        **quant_args,
    )
    return model, tokenizer


def build_peft(model, cfg: TrainingConfig):
    """Create / attach a LoRA adapter with PEFT only."""
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def prepare_dataset(cfg: TrainingConfig):
    rows = load_jsonl(cfg.training_file)
    if cfg.loss == "sft":
        ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    else:
        ds = Dataset.from_list(rows)

    if cfg.test_file:
        rows_test = load_jsonl(cfg.test_file)
        if cfg.loss in {"orpo", "dpo"}:
            test_ds = Dataset.from_list(rows_test)
        else:
            test_ds = Dataset.from_list([{"messages": r["messages"]} for r in rows_test])
    else:
        split = ds.train_test_split(test_size=0.1, seed=cfg.seed)
        ds, test_ds = split["train"], split["test"]

    return ds, test_ds


def sft_train(cfg, train_ds, model, tokenizer, test_ds):
    """
    Very lightweight equivalent of your old `sft_train`, using vanilla Trainer.
    It just tokenizes `messages` and does next-token LM.
    """
    def tokenize(example):
        text = example["messages"] if isinstance(example["messages"], str) else json.dumps(example["messages"])
        tokens = tokenizer(text, truncation=True, max_length=cfg.max_seq_len)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_tok = train_ds.map(tokenize, batched=False, num_proc=1, remove_columns=train_ds.column_names)
    test_tok  = test_ds.map(tokenize,  batched=False, num_proc=1, remove_columns=test_ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        fp16=not cfg.bf16 and torch.cuda.is_available(),
        bf16=cfg.bf16 and torch.cuda.is_available(),
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=cfg.seed,
        max_grad_norm=1.0,
    )
    return Trainer(model, args, train_dataset=train_tok, eval_dataset=test_tok, data_collator=collator)


# ---------------------------------------------------------------------- #
#                            MAIN PIPELINE                               #
# ---------------------------------------------------------------------- #
def train(cfg: TrainingConfig):
    model, tok = load_model_and_tokenizer(cfg.model, load_in_4bit=cfg.load_in_4bit)
    model = build_peft(model, cfg)
    train_ds, test_ds = prepare_dataset(cfg)

    trainer = sft_train(cfg, train_ds, model, tok, test_ds)
    trainer.train()

    # ─── push to hub ────────────────────────────────────────────────────
    push_model(cfg, cfg.finetuned_model_id, model, tok)

    try:
        print(trainer.evaluate())
    except Exception as e:
        print(f"Could not evaluate: {e}")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(cfg, repo, model, tok):
    if cfg.merge_before_push:
        model.merge_and_unload()        # full-precision merge
        model.push_to_hub(repo, token=os.environ["HF_TOKEN"], private=cfg.push_to_private)
        tok.push_to_hub(repo,   token=os.environ["HF_TOKEN"], private=cfg.push_to_private)
    else:
        model.push_to_hub(repo, token=os.environ["HF_TOKEN"], private=cfg.push_to_private)
        tok.push_to_hub(repo,   token=os.environ["HF_TOKEN"], private=cfg.push_to_private)


def main(path):
    with open(path) as f:
        cfg_dict = json.load(f)
    train(TrainingConfig(**cfg_dict))


if __name__ == "__main__":
    main(sys.argv[1])
