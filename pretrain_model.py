import argparse
import random
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from pretrain_dataset import LcquadPretrainingDataset

def fix_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Pretraining Pipeline for Qwen2.5-Coder Instruct")
    parser.add_argument("--input_file", type=str, required=True, help="Path to LC-QUAD 2.0 training JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the resulting model")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--logging_steps", type=int, default=1000)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set random seed for reproducibility.
    fix_seed(args.seed)

    # 1. Initialize model and tokenizer.
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    new_rubq_tokens = ["```"] + ['wdt:', 'skos:', 'wd:', 'p:', 'ps:', 'pq:'] + ["select", "where", "filter", "optional", "distinct",
                       "group", "order", "limit", "offset", "prefix", "p", "q", "now", "asc", "desc", "ask", "count"]
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_tokens(new_rubq_tokens)
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare response template IDs and collator.
    response_template_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # 2. Load dataset.
    with open(args.input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)['dataset']
    train_dataset = LcquadPretrainingDataset(samples, tokenizer, max_length=args.max_length)

    # 3. Define training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        report_to='none'
    )

    # 4. Initialize SFTTrainer.
    sft_trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # 5. Start training.
    sft_trainer.train()
    sft_trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
