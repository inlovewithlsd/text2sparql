import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import preprocess_sparql, mask_query, add_extra_relations, add_extra_entities, load_wikidata_entities, load_wikidata_relations

# Global instructions dictionary.
INSTRUCTIONS = {
    'en': """You are an expert SPARQL query generator for Wikidata. Your task is to transform natural language questions into correct and efficient SPARQL queries, ensuring precise alignment with the question, the provided entities, relations, and valid triplets.You are given valid Wikidata triplets constructed from the provided entities and relations. These triplets follow the correct syntax and use the proper prefixes. Provided vlaid triplets are based solely on the given information; if the query requires connections or entities not explicitly provided, generate new triplets accordingly while maintaining valid Wikidata syntax.
            Ensure proper structure and syntax of SPARQL query. Optimize queries for performance, applying filters, counts, and conditions when necessary. Output only the complete SPARQL query with correct formatting, without explanations or extra text. Handle missing entities, ambiguous cases, and complex queries logically. When multiple entities or relations are provided, select the one that best aligns with the question context and generate a query that includes only that selection.
            Ensure the syntax correctness of generated SPARQL query. Double-check that queries have correctly closed parentheses and braces.
        """
}

def format_entities(entity_map):
    if not entity_map:
        return " - None\n"
    return "".join(f"[{qid}] - ({label.get('en', 'N/A')})\n" for qid, label in entity_map.items() if label)

def format_predicates(relation_map):
    if not relation_map:
        return " - None\n"
    return "".join(f"[{pid}] - ({label.get('en', 'N/A')})\n" for pid, label in relation_map.items() if label)

def create_prompt(question, entities_string, predicates_string, valid_triplets=None):
    # valid_triplets is currently unused; can be extended as needed.
    return f"Question: {question}\n\nEntities:\n{entities_string}\n\nRelations:\n{predicates_string}\n"

def format_dataset(dataset, tokenizer, mode='train', lang='en'):
    sft_examples_list, failed_samples = [], []
    instruction = INSTRUCTIONS[lang]

    entities_lookup = load_wikidata_entities()
    relations_lookup = load_wikidata_relations()

    for sample in tqdm(dataset, desc="Formatting dataset"):
        question = sample.get(f'{lang}_question', "").strip()
        if not question:
            failed_samples.append(sample)
            continue

        # Get entities and relations from the sample.
        entity_map = sample.get('entities', {}).get('question') or sample.get('entities', {}).get('query') or {}
        relation_map = sample.get('relations', {}).get('question') or sample.get('relations', {}).get('query') or {}

        entities_string = format_entities(add_extra_entities(entity_map, entities_lookup))
        predicates_string = format_predicates(add_extra_relations(relation_map, relations_lookup))

        user_task = create_prompt(question, entities_string, predicates_string)

        sparql = preprocess_sparql(sample.get('query', ""))
        # sparql = mask_query(sparql)
        target = f"```\n{sparql}\n```"

        if not entities_string.strip() or not target:
            failed_samples.append(sample)
            continue

        if mode == 'train':
            chat = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_task},
                {"role": "assistant", "content": target}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        else:
            chat = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_task}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        sft_examples_list.append({"id": str(sample.get("id")), "sft": formatted_prompt, "sparql": sparql})
    return sft_examples_list, failed_samples

def main():
    parser = argparse.ArgumentParser(description="Create SFT dataset for SPARQL pretraining.")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Path or model name for the tokenizer")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the SFT datasets")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name to use as output file prefix")
    parser.add_argument("--lang", type=str, default="en", help="Language (default: en)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Process train dataset.
    train_data = json.load(open(args.train_file, "r", encoding="utf-8"))["dataset"]
    train_examples, train_failed = format_dataset(train_data, tokenizer, mode="train", lang=args.lang)
    train_out_path = os.path.join(args.output_dir, f"{args.dataset_name}_train.json")
    with open(train_out_path, "w", encoding="utf-8") as f:
        json.dump(train_examples, f, ensure_ascii=False, indent=4)

    # Process test dataset.
    test_data = json.load(open(args.test_file, "r", encoding="utf-8"))["dataset"]
    test_examples, test_failed = format_dataset(test_data, tokenizer, mode="test", lang=args.lang)
    test_out_path = os.path.join(args.output_dir, f"{args.dataset_name}_test.json")
    with open(test_out_path, "w", encoding="utf-8") as f:
        json.dump(test_examples, f, ensure_ascii=False, indent=4)

    print("Prepared SFT train samples:", len(train_examples))
    print("Total train failed samples:", len(train_failed))
    print("Prepared SFT test samples:", len(test_examples))
    print("Total test failed samples:", len(test_failed))

if __name__ == "__main__":
    main()
