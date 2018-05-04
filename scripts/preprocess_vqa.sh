#! /bin/bash

####   Creating vocabulary   ##### 
# Create directory for extracted vocabulary
if [ ! -d "data/VQA_v2.0/preprocess/vocabulary" ]; then
  mkdir -p data/VQA_v2.0/preprocess/vocabulary
fi

# Construct vocabulary for training and validation dataset of VQA v1.9
echo "=====> Constructing vocabulary on train/val split"
python -m src.preprocess.construct_vocabulary \
	--vocab_option 'raw' \
	--target_splits 'train' 'val' \
	--save_vocab_dir 'data/VQA_v2.0/preprocess/vocabulary' \
	--dataset 'vqa' \
	--dataset_config_path './data/vqa_v2.0_path.yml' \
	--threshold_num_answer 3000



#####   Encoding question and answers   ##### 
# Preprocess question and answers and save tokenized results.
if [ ! -d "data/VQA_v2.0/preprocess/encoded_qa" ]; then
	mkdir -p data/VQA_v2.0/preprocess/encoded_qa
fi

# Preprocessing train split
echo $'\n=====> Preprocessing on train split'
stdbuf -oL python -m src.preprocess.preprocess_vqa \
	--target_splits 'train' \
	--vocab_path 'data/VQA_v2.0/preprocess/vocabulary/vqa_3000_vocab_train-val_raw.json' \
	--save_encoded_qa_dir 'data/VQA_v2.0/preprocess/encoded_qa' \
	--question_filter_option 'all_questions_with_answer_vocab' \
	2>&1 | tee logs/log_encode_question_answer_VQA_v2.0_train_3000.log

# Preprocessing val split
echo $'\n=====> Preprocessing on val split'
stdbuf -oL python -m src.preprocess.preprocess_vqa \
	--target_splits 'val' \
	--vocab_path 'data/VQA_v2.0/preprocess/vocabulary/vqa_3000_vocab_train-val_raw.json' \
	--save_encoded_qa_dir 'data/VQA_v2.0/preprocess/encoded_qa' \
	--question_filter_option 'all_questions_with_answer_vocab' \
	2>&1 | tee logs/log_encode_question_answer_VQA_v2.0_val_3000.log

# Preprocessing train-val split
echo $'\n=====> Preprocessing on train/val split'
stdbuf -oL python -m src.preprocess.preprocess_vqa \
	--target_splits 'train' 'val' \
	--vocab_path 'data/VQA_v2.0/preprocess/vocabulary/vqa_3000_vocab_train-val_raw.json' \
	--save_encoded_qa_dir 'data/VQA_v2.0/preprocess/encoded_qa' \
	--question_filter_option 'all_questions_with_answer_vocab' \
	2>&1 | tee logs/log_encode_question_answer_VQA_v2.0_train-val_3000.log

# Preprocessing test split
echo $'\n=====> Preprocessing on test split'
stdbuf -oL python -m src.preprocess.preprocess_vqa \
	--target_splits 'test' \
	--vocab_path 'data/VQA_v2.0/preprocess/vocabulary/vqa_3000_vocab_train-val_raw.json' \
	--save_encoded_qa_dir 'data/VQA_v2.0/preprocess/encoded_qa' \
	--question_filter_option 'only_questions' \
	2>&1 | tee logs/log_encode_question_answer_VQA_v2.0_test_from_train-val_3000_vocab.log

# Preprocessing test split
echo $'\n=====> Preprocessing on test split'
stdbuf -oL python -m src.preprocess.preprocess_vqa \
	--target_splits 'test-dev' \
	--vocab_path 'data/VQA_v2.0/preprocess/vocabulary/vqa_3000_vocab_train-val_raw.json' \
	--save_encoded_qa_dir 'data/VQA_v2.0/preprocess/encoded_qa' \
	--question_filter_option 'only_questions' \
	2>&1 | tee logs/log_encode_question_answer_VQA_v2.0_test-dev_from_train-val_3000_vocab.log
