#python -m src.preprocess.preprocess_clevr \
#	--target_splits train \
#	--vocab_path data/CLEVR_v1.0/preprocess/vocabulary/clevr_vocab_train_raw.json \
#	--use_zero_token \
#	--max_question_length 45 \
#
#python -m src.preprocess.preprocess_clevr \
#	--target_splits val \
#	--vocab_path data/CLEVR_v1.0/preprocess/vocabulary/clevr_vocab_train_raw.json \
#	--use_zero_token \
#	--max_question_length 45 \

python -m src.preprocess.preprocess_clevr \
	--target_splits test \
	--vocab_path data/CLEVR_v1.0/preprocess/vocabulary/clevr_vocab_train_raw.json \
	--use_zero_token \
	--max_question_length 45 \

#####################################################################################
#python -m src.preprocess.preprocess_clevr \
#	--target_splits train \
#	--vocab_path data/CLEVR_v1.0/preprocess/vocabulary/clevr_vocab_train_raw.json \
#	--use_zero_token \
#	--max_question_length 45 \
#	--question_family_index_list 29 33 52 75 81 88 # color
#
#python -m src.preprocess.preprocess_clevr \
#	--target_splits val \
#	--vocab_path data/CLEVR_v1.0/preprocess/vocabulary/clevr_vocab_train_raw.json \
#	--use_zero_token \
#	--max_question_length 45 \
#	--question_family_index_list 29 33 52 75 81 88 # color
#
#python -m src.preprocess.preprocess_clevr \
#	--target_splits train \
#	--vocab_path data/CLEVR_v1.0/preprocess/vocabulary/clevr_vocab_train_raw.json \
#	--use_zero_token \
#	--max_question_length 45 \
#	--question_family_index_list 29 33 52 75 81 88 30 34 53 76 82 87 # color + material
