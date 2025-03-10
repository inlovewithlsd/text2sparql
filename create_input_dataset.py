import json
import argparse
from transformers import AutoTokenizer
from utils import format_qid, create_prompt, INSTRUCTIONS
def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")

def process_dataset(tokenizer, entities_path, predicates_path, output_path):
    entity_candidates = load_json(entities_path)
    predicate_candidates = load_json(predicates_path)

    ids = set(entity_candidates.keys()) & set(predicate_candidates.keys())
    dataset = []

    for qid in ids:
        entity_data = entity_candidates[qid]
        relation_data = predicate_candidates[qid]

        assert entity_data["question_eng"] == relation_data["question_eng"], "Mismatched questions"
        assert entity_data["query"] == relation_data["query"], "Mismatched queries"

        entities_string = format_qid(entity_data["candidates"])
        predicates_string = format_qid(relation_data["candidates"])
        user_task = create_prompt(entity_data["question_eng"], entities_string, predicates_string)
        chat = [
            {"role": "system", "content": INSTRUCTIONS['en']},
            {"role": "user", "content": user_task}
        ]
        formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        dataset.append({
            "id": qid,
            "question": entity_data["question_eng"],
            "query": entity_data["query"],
            "entities": entity_data["candidates"],
            "relations": relation_data["candidates"],
            "sft": formatted_prompt
        })


    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process entity and relation candidates into a structured dataset.")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Path or model name for the tokenizer")
    parser.add_argument("--entities", required=True, help="Path to entity candidates JSON file")
    parser.add_argument("--predicates", required=True, help="Path to predicate (relation) candidates JSON file")
    parser.add_argument("--output", required=True, help="Output path for processed dataset JSON file")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    process_dataset(tokenizer, args.entities, args.predicates, args.output)