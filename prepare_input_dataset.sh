python3 create_input_dataset.py \
 --tokenizer_path "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
 --entities "wikidata_candidates/rubq_result_entity_10.json" \
 --predicates "wikidata_candidates/rubq_result_property_10.json" \
 --output "data/rubq_input_dataset.json"