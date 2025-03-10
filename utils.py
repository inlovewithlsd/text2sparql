import re
import random

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

def corrupt_sparql(sparql):
    tokens = sparql.split()
    if len(tokens) > 5:
        remove_idx = random.randint(0, len(tokens) - 1)
        corrupted_tokens = tokens[:remove_idx] + tokens[remove_idx + 1:]
        return " ".join(corrupted_tokens)
    return sparql


def mix_triplets(sparql):
    match = re.search(r'\{(.*)\}', sparql, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        # Split on '.' to get candidate triplet segments.
        triplets = [t.strip() for t in inner.split('.') if t.strip()]
        new_triplets = []
        for trip in triplets:
            tokens = trip.split()
            if len(tokens) >= 3:
                main = tokens[:3]
                random.shuffle(main)
                new_trip = " ".join(main + tokens[3:])
            else:
                new_trip = trip
            new_triplets.append(new_trip)
        new_inner = " . ".join(new_triplets)
        new_sparql = sparql[:match.start()] + "{ " + new_inner + " }" + sparql[match.end():]
        return new_sparql
    return sparql

def add_extra_brackets(sparql):
    tokens = sparql.split()
    brackets = ["(", ")", "{", "}"]
    bracket = random.choice(brackets)
    idx = random.randint(0, len(tokens))
    tokens.insert(idx, bracket)
    return " ".join(tokens)

def replace_entity(sparql: str) -> str:
    tokens = sparql.split()
    for i, token in enumerate(tokens):
        if token.startswith('wd:') or token.startswith('wdt:') or token.startswith('p:') or token.startswith('ps:') or token.startswith('pq:'):
            # Replace the token with a placeholder, preserving the prefix.
            prefix = token.split(':')[0]
            tokens[i] = f"{prefix}:{random.randint(0, 10**5)}"
    return " ".join(tokens)


def augment_sparql(sparql):
    augmentation_types = ['remove_token', 'replace_entity', 'add_extra_brackets', 'mix_triplets', 'corrupt']
    chosen = random.choice(augmentation_types)

    if chosen == 'remove_token':
        return corrupt_sparql(sparql)
    elif chosen == 'replace_entity':
        return replace_entity(sparql)
    elif chosen == 'add_extra_brackets':
        return add_extra_brackets(sparql)
    elif chosen == 'mix_triplets':
        return mix_triplets(sparql)
    elif chosen == 'corrupt':
        return corrupt_sparql(sparql)
    else:
        return sparql


def mask_query(query):
    entity_mapping = {}
    relation_mapping = {}
    entity_count = 0
    relation_count = 0

    # Function to replace an entity match with a masked token.
    def replace_entity(match):
        nonlocal entity_count
        entity = match.group(0)
        if entity not in entity_mapping:
            entity_mapping[entity] = f"<entity_{entity_count}>"
            entity_count += 1
        return entity_mapping[entity]

    def replace_relation(match):
        nonlocal relation_count
        relation = match.group(0)
        if relation not in relation_mapping:
            relation_mapping[relation] = f"<relation_{relation_count}>"
            relation_count += 1
        return relation_mapping[relation]

    masked_query = re.sub(r'Q\d+', replace_entity, query)
    masked_query = re.sub(r'P\d+', replace_relation, masked_query)

    return masked_query

def validate_corruptions(sparql):
    print('Original sparql:', sparql, end='\n\n')
    print("remove_token", corrupt_sparql(sparql))
    print("replace_entity", replace_entity(sparql))
    print("add_extra_brackets", add_extra_brackets(sparql))
    print("mix_triplets", mix_triplets(sparql))
    print("augment_sparql", augment_sparql(sparql))


if __name__ == "__main__":
    example_sparql = "select ?answer where { wd:Q8070 wdt:P828 ?answer }"
    preprocessed_sparql = preprocess_sparql(example_sparql)
    validate_corruptions(preprocessed_sparql)