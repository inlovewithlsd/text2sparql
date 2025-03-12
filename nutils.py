import json
import re
import json
import random

INSTRUCTION = """You are an expert SPARQL query generator for Wikidata. Your task is to transform natural language questions into correct and efficient SPARQL queries, ensuring precise alignment with the question, the provided entities, relations, and valid triplets.You are given valid Wikidata triplets constructed from the provided entities and relations. These triplets follow the correct syntax and use the proper prefixes. Provided vlaid triplets are based solely on the given information; if the query requires connections or entities not explicitly provided, generate new triplets accordingly while maintaining valid Wikidata syntax.
Ensure proper structure and syntax of SPARQL query. Optimize queries for performance, applying filters, counts, and conditions when necessary. Output only the complete SPARQL query with correct formatting, without explanations or extra text. Handle missing entities, ambiguous cases, and complex queries logically. When multiple entities or relations are provided, select the one that best aligns with the question context and generate a query that includes only that selection.
Ensure the syntax correctness of generated SPARQL query. Double-check that queries have correctly closed parentheses and braces."""


def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")

def format_gold_qid(id_map):
    if not id_map:
        return " - None\n"
    return "".join(f"{pid} - {label.get('en', 'N/A')}\n" for pid, label in id_map.items() if label)


def format_qid(id_map):
    if not id_map:
        return " - None\n"
    return "".join(f"{qid} - {label}\n" for qid, label in id_map.items() if label)

def format_refined_entities(refined_entities):
    return {
        entity['id']: {
            'en': entity['label']
        }
        for entity in refined_entities
    }

def create_prompt(question, entities_string, predicates_string):
    return f"Question: {question}\n\nEntities:\n{entities_string}\n\nRelations:\n{predicates_string}\n"


def load_wikidata_entities(entities_file='data/wikidata_files/wikidata_entities.json'):
    with open(entities_file, 'r', encoding='utf-8') as f:
        entities_list = json.load(f)
    return {entity["id"]: entity for entity in entities_list if "id" in entity}

def load_wikidata_relations(relations_file='data/wikidata_files/wikidata_relations.json'):
    with open(relations_file, 'r', encoding='utf-8') as f:
        relations = json.load(f)
    return relations


import random
import re


def generate_alternative_label(gold_label):
    if len(gold_label) < 4:
        return gold_label
    removal_length = random.randint(1, max(len(gold_label) - 3, 2))
    shortened = gold_label[:-removal_length]
    alternatives = [gold_label, shortened, f"{gold_label} (alt)", f"first {gold_label}" ]
    return random.choice(alternatives)


def get_candidate_label(candidate_id, gold_label, records):
    record = records.get(candidate_id)
    if record:
        aliases = record.get("aliases")
        if isinstance(aliases, list) and aliases:
            return random.choice(aliases)
        elif isinstance(aliases, str) and aliases.strip():
            return aliases.strip()
        elif record.get("label"):
            return record["label"]
    return generate_alternative_label(gold_label)


def generate_extra_item(gold_id, gold_label, existing_ids, records, prefix):
    pattern = rf'{prefix}(\d+)$'
    match = re.match(pattern, gold_id)
    if not match:
        return None, None
    num = int(match.group(1))
    offsets = [-3, -2, -1, 1, 2, 3]
    random.shuffle(offsets)
    for offset in offsets:
        new_num = num + offset
        if new_num <= 0:
            continue
        candidate_id = f"{prefix}{new_num}"
        if candidate_id not in existing_ids:
            candidate_label = get_candidate_label(candidate_id, gold_label, records)
            return candidate_id, candidate_label
    return None, None


def add_extra_items(gold_items, records, prefix):
    extra_items = {}
    n_extra = random.randint(0, 3)
    gold_ids = list(gold_items.keys())
    # Use a combined dictionary of existing gold and extra items to avoid duplicates.
    combined_ids = {**gold_items, **extra_items}

    for _ in range(n_extra):
        if not gold_ids:
            break
        gold_id = random.choice(gold_ids)
        gold_label = gold_items[gold_id].get("en", "unknown")
        candidate_id, candidate_label = generate_extra_item(gold_id, gold_label, combined_ids, records, prefix)
        if candidate_id and candidate_label:
            extra_items[candidate_id] = {"en": candidate_label}
            combined_ids[candidate_id] = True  # add to combined_ids to avoid reuse
    merged = gold_items.copy()
    merged.update(extra_items)
    return merged


def add_extra_entities(entities, wikidata_entities):
    return add_extra_items(entities, wikidata_entities, prefix="Q")


def add_extra_relations(relations, wikidata_relations):
    return add_extra_items(relations, wikidata_relations, prefix="P")

def is_schema_token(token, prefixes):
    # Check if token starts with any of the given prefixes.
    return any(token.startswith(prefix) for prefix in prefixes)

def preprocess_sparql(sparql: str) -> str:
    # Preprocess a SPARQL query to help tokenization and enforce a standard format.
    sparql = sparql.replace('SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }', '')

    sparql = sparql.replace('\n', ' ')
    sparql = sparql.replace('{', ' { ')
    sparql = sparql.replace('}', ' } ')
    sparql = sparql.replace('(', ' ( ')
    sparql = sparql.replace(')', ' ) ')
    sparql = sparql.replace('[', ' [ ')
    sparql = sparql.replace(']', ' ] ')
    sparql = sparql.replace(',', ' , ')
    sparql = sparql.replace('.', ' . ')
    sparql = sparql.replace('|', ' | ')
    sparql = sparql.replace('/', ' / ')
    sparql = sparql.replace(';', ' ; ')

    sparql = sparql.strip()
    sparql_tokens = sparql.split()
    updated_tokens = []
    # Lowercase non-schema tokens.
    for token in sparql_tokens:
        token = token.strip()
        if not is_schema_token(token, ['dr:', 'wd:', 'wdt:', 'p:', 'pq:', 'ps:', 'psn:']):
            updated_tokens.append(token.lower())
        else:
            updated_tokens.append(token)

    updated_sparql = " ".join(updated_tokens).strip()
    updated_sparql = updated_sparql.replace('. }', ' }')
    return updated_sparql