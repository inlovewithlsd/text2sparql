python prepare_validation_sft_dataset.py \
    --dataset "data/datasets/rubq/rubq_test_we.json" \
    --tokenizer_path "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --entities "data/miron_entities/rubq_result_entity_10.json" \
    --predicates "data/miron_entities/rubq_result_property_10.json" \
    --output "data/e2e_validation_datasets/rubq_input_dataset.json" \
    --mode "miron"