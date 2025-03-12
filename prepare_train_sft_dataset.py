import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from nutils import INSTRUCTION, preprocess_sparql, create_prompt, load_wikidata_entities, load_wikidata_relations, format_gold_qid, add_extra_entities, add_extra_relations

def format_dataset(dataset, tokenizer, entities_file, relations_file, mode='e2e', phase='train', lang='en'):
    sft_examples_list, failed_samples = [], []
    instruction = INSTRUCTION

    wikidata_entities = load_wikidata_entities(entities_file)
    wikidata_relations = load_wikidata_relations(relations_file)

    for sample in tqdm(dataset, desc="Formatting dataset"):
        question = sample.get(f'{lang}_question', "").strip()
        if not question:
            failed_samples.append(sample)
            continue

        # Get entities and relations from the sample.
        entity_map = sample.get('entities', {}).get('question') or sample.get('entities', {}).get('query') or {}
        relation_map = sample.get('relations', {}).get('question') or sample.get('relations', {}).get('query') or {}

        if mode == 'e2e':
            entities_string = format_gold_qid(add_extra_entities(entity_map, wikidata_entities))
            predicates_string = format_gold_qid(add_extra_relations(relation_map, wikidata_relations))
        else:
            entities_string = format_gold_qid(entity_map)
            predicates_string = format_gold_qid(relation_map)

        user_task = create_prompt(question, entities_string, predicates_string)

        query = preprocess_sparql(sample.get('query', ""))
        multi_hop_query = preprocess_sparql(sample.get('multi_hop_query')) if sample.get('multi_hop_query') is not None else None

        intermediate_entities = sample.get('intermediate_entities', [])
        intermediate_entities = [] if not intermediate_entities else intermediate_entities

        if multi_hop_query:
            replacements_made = False
            for entity in intermediate_entities:
                if f"wd:{entity}" in multi_hop_query:
                    multi_hop_query = multi_hop_query.replace(f"wd:{entity}", "<|mask|>")
                    replacements_made = True

            if replacements_made:
                target = f'multi-hop ```{query}``` <|sep|> ```{multi_hop_query}```'
                sparql = sample.get('multi_hop_query')
            else:
                target = f'single-hop ```{multi_hop_query}```'
                sparql = sample.get('multi_hop_query')
        else:
            target = f'single-hop ```{query}```'
            sparql = sample.get('query')

        if not entities_string.strip() or not query:
            failed_samples.append(sample)
            continue

        if not entities_string.strip() or not target:
            failed_samples.append(sample)
            continue

        if phase == 'train':
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
    parser = argparse.ArgumentParser(description="Create SFT dataset for SPARQL training.")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Path or model name for the tokenizer")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--mode", type=str, required=True, help="Training mode: e2e or text2sparql")
    parser.add_argument("--entities_file", type=str, required=True, help="Path to wikidata entities")
    parser.add_argument("--relations_file", type=str, required=True, help="Path to wikidata relations")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the SFT datasets")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name to use as output file prefix")
    parser.add_argument("--lang", type=str, default="en", help="Language (default: en)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Process train dataset.
    train_data = json.load(open(args.train_file, "r", encoding="utf-8"))["dataset"]
    train_examples, train_failed = format_dataset(train_data, tokenizer, args.entities_file, args.relations_file, mode=args.mode, phase="train", lang=args.lang)
    train_out_path = os.path.join(args.output_dir, f"{args.dataset_name}_train.json")
    with open(train_out_path, "w", encoding="utf-8") as f:
        json.dump(train_examples, f, ensure_ascii=False, indent=4)

    # Process test dataset.
    test_data = json.load(open(args.test_file, "r", encoding="utf-8"))["dataset"]
    test_examples, test_failed = format_dataset(test_data, tokenizer, args.entities_file, args.relations_file, mode=args.mode, phase="test", lang=args.lang)
    test_out_path = os.path.join(args.output_dir, f"{args.dataset_name}_test.json")
    with open(test_out_path, "w", encoding="utf-8") as f:
        json.dump(test_examples, f, ensure_ascii=False, indent=4)

    print("Prepared SFT train samples:", len(train_examples))
    print("Total train failed samples:", len(train_failed))
    print("Prepared SFT test samples:", len(test_examples))
    print("Total test failed samples:", len(test_failed))

if __name__ == "__main__":
    main()