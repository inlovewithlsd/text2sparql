#!/usr/bin/env python
import asyncio
import aiohttp
import json
import re
import pandas as pd
from tqdm import tqdm

SEM = asyncio.Semaphore(10)

async def execute_sparql(session, query, timeout=10, max_retries=3):
    if not query:
        return None
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    data = {"query": query, "format": "json"}
    async with SEM:
        for attempt in range(max_retries):
            try:
                async with session.post(url, data=data, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        results = await response.json()
                        return extract_answers(results)
                    elif response.status == 400:
                        return None
            except Exception:
                if attempt == max_retries - 1:
                    return []
                await asyncio.sleep(1)
    return []

def extract_answers(resp):
    answers = []
    if "results" in resp:
        for binding in resp["results"]["bindings"]:
            for key, val in binding.items():
                value = val.get("value", "")
                # Extract wikidata id if it matches.
                if re.match(r"https?://www\.wikidata\.org/entity/Q\d+", value):
                    answers.append(value.split("/")[-1])
                else:
                    answers.append(value)
    elif "boolean" in resp:
        answers.append(resp["boolean"])
    return answers

def extract_sparql(text: str) -> str:
    pattern = r"```(?:[^\n]*\n)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()

def calculate_metrics(correct, predicted):
    c_set, p_set = set(correct), set(predicted)
    em = c_set == p_set
    tp = len(c_set & p_set)
    precision = tp / len(p_set) if p_set else 0
    recall = tp / len(c_set) if c_set else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    return {"em": em, "f1": f1, "precision": precision, "recall": recall}

async def main():
    finetuned_results = pd.read_json("inference_results/pat_inference.json")
    metrics_list = []
    async with aiohttp.ClientSession() as session:
        for _, row in tqdm(finetuned_results.iterrows(), total=finetuned_results.shape[0], desc="Processing"):
            gold_query = row['gold_query']
            pred_query = extract_sparql(row['predicted_query'])

            gold_entities = await execute_sparql(session, gold_query)
            pred_entities = await execute_sparql(session, pred_query)
            print(pred_entities)

            if not gold_entities:
                continue
            if pred_entities is None:
                metric = {"em": False, "f1": 0, "precision": 0, "recall": 0, "incorrect": True, "empty": False}
            else:
                metric = calculate_metrics(gold_entities, pred_entities)
                metric.update({"incorrect": False, "empty": len(pred_entities) == 0})
            metric.update({"id": row["id"], "dataset": row.get("dataset", "")})
            metrics_list.append(metric)
    with open("qwen_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, ensure_ascii=False, indent=4)
    avg_bleu = sum(m.get("f1", 0) for m in metrics_list) / len(metrics_list) if metrics_list else 0.0
    print(f"Average F1 score: {avg_bleu:.4f}")
    print("Metrics saved to qwen_baseline_metrics.json")

if __name__ == "__main__":
    asyncio.run(main())
