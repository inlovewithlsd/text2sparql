import json

wikidata_graph = json.load(open('data/all_predicates_data.json'))
dataset = json.load(open('data/rubq_input_dataset.json'))

question = dataset[0]

print(question)