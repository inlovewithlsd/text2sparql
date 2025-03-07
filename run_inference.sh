#!/bin/bash
python inference_model.py \
  --input_file "sft/rubq_test.json" \
  --model_name_or_path "./drive/MyDrive/text2sparql/models/rubq_model/" \
  --output_file "./drive/MyDrive/text2sparql/inference_results/rubq_inference.json" \
  --max_length 1024 \
  --batch_size 8 \
  --seed 42