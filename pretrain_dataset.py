import torch
from torch.utils.data import Dataset
from utils import preprocess_sparql, augment_sparql

instruction = (
    "Given the natural language question with its associated entities and relations, "
    "correct the corrupted SPARQL query below by fixing any syntax errors or misplaced tokens."
)

class LcquadPretrainingDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=768):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        gold_sparql = sample.get("query", "")
        preprocessed_sparql = preprocess_sparql(gold_sparql)

        corrupted_sparql = augment_sparql(preprocessed_sparql)
        target = f"```\n{preprocessed_sparql}\n```"

        # Build entity and relation mappings.
        entity_map = (
            sample.get('entities', {}).get('question')
            or sample.get('entities', {}).get('query')
            or {}
        )
        relation_map = (
            sample.get('relations', {}).get('question')
            or sample.get('relations', {}).get('query')
            or {}
        )

        entities_str = (
            "".join(
                f"[{qid}] - ({label.get('en', 'N/A')})\n"
                for qid, label in entity_map.items() if label
            )
            if entity_map else " - None\n"
        )
        relations_str = (
            "".join(
                f"[{pid}] - ({label.get('en', 'N/A')})\n"
                for pid, label in relation_map.items() if label
            )
            if relation_map else " - None\n"
        )

        # Use the English question for the prompt.
        question = sample.get("en_question", "")

        # Build the user prompt with the question, mappings, and corrupted SPARQL.
        user_task = (
            f"Question: {question}\n"
            f"Entities:\n{entities_str}\n"
            f"Relations:\n{relations_str}\n"
            f"SPARQL: {corrupted_sparql}"
        )

        # Construct the SFT chat template.
        chat = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_task},
            {"role": "assistant", "content": target}
        ]

        # Format the prompt using the tokenizer's chat template.
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize the prompt.
        tokenized_sft = self.tokenizer(
            formatted_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_tensors='pt'
        )

        input_ids = tokenized_sft['input_ids'][0].to(torch.long)
        attention_mask = tokenized_sft['attention_mask'][0].to(torch.bool)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }