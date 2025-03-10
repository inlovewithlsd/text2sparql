INSTRUCTIONS = {
    'en': """You are an expert SPARQL query generator for Wikidata. Your task is to transform natural language questions into correct and efficient SPARQL queries, ensuring precise alignment with the question, the provided entities, relations, and valid triplets.You are given valid Wikidata triplets constructed from the provided entities and relations. These triplets follow the correct syntax and use the proper prefixes. Provided vlaid triplets are based solely on the given information; if the query requires connections or entities not explicitly provided, generate new triplets accordingly while maintaining valid Wikidata syntax.
            Ensure proper structure and syntax of SPARQL query. Optimize queries for performance, applying filters, counts, and conditions when necessary. Output only the complete SPARQL query with correct formatting, without explanations or extra text. Handle missing entities, ambiguous cases, and complex queries logically.
            Ensure the syntax correctness of generated SPARQL query. Double-check that queries have correctly closed parentheses and braces.
        """
}

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

def format_qid(entity_map):
    if not entity_map:
        return " - None\n"
    return "".join(f"[{qid}] - ({label})\n" for qid, label in entity_map.items() if label)

def create_prompt(question, entities_string, predicates_string, valid_triplets=None):
    # valid_triplets is currently unused; can be extended as needed.
    return f"Question: {question}\n\nEntities:\n{entities_string}\n\nRelations:\n{predicates_string}\n"
