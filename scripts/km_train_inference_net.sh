
# running command and its configuration path
#bash scripts/km_train_model.sh IE_prob infer vqa 1 2 0 # src/experiment/options/vqa/infer/IE_prob.yml
#bash scripts/train_model.sh IE_logit infer vqa 1 2 0 # src/experiment/options/vqa/infer/IE_logit.yml
bash scripts/km_train_model.sh KD-MCL_prob infer vqa 3 2 0 # src/experiment/options/vqa/infer/KD-MCL_prob.yml
#bash scripts/train_model.sh KD-MCL_logit infer vqa 3 2 0 # src/experiment/options/vqa/infer/KD-MCL_prob.yml

