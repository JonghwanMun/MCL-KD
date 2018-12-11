# Experiments

### Training VQA Network using MCL-KD

You can train VQA Network using MCL-KD with following command or running `train_model.sh` in script folder.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.train \
	--config_path src/experiment/options/clevr/ensemble/KD-MCL-saaa/default.yml \
	--model_type ensemble \
	--dataset clevr \
	--num_workers 4
```
Then, training outputs are saved in `results/dataset/model_type/option_path`.

### Evaluating VQA Network

You can evaluate VQA Network with following command or running `eval_model.sh` in script folder.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval \
	--exp KD-MCL-saaa/default \
	--model_type ensemble \
	--dataset clevr \
	--start_epoch 0 \
	--end_epoch 100 \
	--epoch_stride 5 \
	--num_workers 4 \
```
