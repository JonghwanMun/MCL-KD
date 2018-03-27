#! /bin/bash

####   Creating vocabulary   ##### 
# Create directory for extracted vocabulary
if [ ! -d "data/CLEVR_v1.0/preprocess/vocabulary" ]; then
  mkdir -p data/CLEVR_v1.0/preprocess/vocabulary
fi

# Construct vocabulary for training and validation dataset of VQA v1.9
echo "=====> Constructing vocabulary on train/val split"
python -m src.preprocess.construct_vocabulary \
	--vocab_option 'raw' \
	--target_splits 'train' \
	--save_vocab_dir 'data/CLEVR_v1.0/preprocess/vocabulary' \
	--dataset 'clevr' \
	--dataset_config_path './data/clevr_path.yml'



#####   Encoding question and answers   ##### 
# Preprocess question and answers and save tokenized results.
if [ ! -d "data/CLEVR_v1.0/preprocess/encoded_qa" ]; then
	mkdir -p data/CLEVR_v1.0/preprocess/encoded_qa
fi

echo $'\n=====> Preprocessing on train split'
python -m src.preprocess.preprocess_clevr \
	--target_splits train \
	--vocab_path data/CLEVR_v1.0/preprocess/vocabulary/clevr_vocab_train_raw.json \
	--use_zero_token \
	--max_question_length 45 \

echo $'\n=====> Preprocessing on validation split'
python -m src.preprocess.preprocess_clevr \
	--target_splits val \
	--vocab_path data/CLEVR_v1.0/preprocess/vocabulary/clevr_vocab_train_raw.json \
	--use_zero_token \
	--max_question_length 45 \

echo $'\n=====> Preprocessing on test split'
python -m src.preprocess.preprocess_clevr \
	--target_splits test \
	--vocab_path data/CLEVR_v1.0/preprocess/vocabulary/clevr_vocab_train_raw.json \
	--use_zero_token \
	--max_question_length 45 \
