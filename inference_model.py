import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from finetune_dataset import SFTFinetuneDataset

def fix_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validate(model, tokenizer, val_dataset, batch_size=8):
    model.eval()
    data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    table = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running Inference"):
            # Move tokenized batch to device.
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                num_beams=2,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
            outputs = outputs.cpu()
            # Assume the prompt length equals the input_ids sequence length.
            input_length = input_ids.shape[1]

            decoded_preds = tokenizer.batch_decode(
                outputs[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # For each prediction in the batch, compute BLEU and record details.
            for i, pred_text in enumerate(decoded_preds):
                gold_query = batch["gold_query"][i]
                pred_query = pred_text.strip()

                table.append({
                    "id": batch["id"][i],
                    "gold_query": gold_query,
                    "predicted_query": pred_query
                })

    return table


def main():
    parser = argparse.ArgumentParser(description="Inference for SPARQL generation using fine-tuned model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the SFT test dataset JSON file")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Pretrained or fine-tuned model name or path")
    parser.add_argument("--output_file", type=str, required=True, help="File path to save the inference results (JSON)")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fix_seed(args.seed)

    # Load model and tokenizer (with FP16 if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Create inference dataset.
    val_dataset = SFTFinetuneDataset(args.input_file, tokenizer, mode="test", max_length=args.max_length)

    # Run inference.
    result_table = validate(model, tokenizer, val_dataset, batch_size=args.batch_size)

    # Save results.
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result_table, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
