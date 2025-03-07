python prepare_sft_datasets.py \
  --train_file "./drive/MyDrive/text2sparql/data/rubq_train.json" \
  --test_file "./drive/MyDrive/text2sparql/data/rubq_test.json" \
  --output_dir "./sft" \
  --dataset_name "rubq" \
  --tokenizer_path "./drive/MyDrive/text2sparql/pretrained_model" \
  --lang "en"