# Experiments

### Training VQA Network using MCL-KD

You can train VQA Network using MCL-KD with following command or running `train_model.sh` in scripts folder.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.train \
	--config_path src/experiment/options/clevr/ensemble/MCL-KD/SAN/m5_k3.yml \
	--model_type ensemble \
	--dataset clevr \
	--num_workers 4
```
Then, training outputs are saved in results/dataset/model_type/config_path, e.g. `results/clevr/ensemble/MCL-KD/SAN/m5_k3/`. <br />
To learn various models presented in the paper, e.g., MCL, CMCL and MCL-KD, refer following commands.
```bash
# MCL-KD with k=1,2,3
bash scripts/train_model.sh MCL-KD/SAN/m5_k1 ensemble clevr 0 2 0
bash scripts/train_model.sh MCL-KD/SAN/m5_k2 ensemble clevr 0 2 0
bash scripts/train_model.sh MCL-KD/SAN/m5_k3 ensemble clevr 0 2 0

# CMCL with k=1,2,3
bash scripts/train_model.sh CMCL/SAN/m5_k1 ensemble clevr 0 2 0
bash scripts/train_model.sh CMCL/SAN/m5_k2 ensemble clevr 0 2 0
bash scripts/train_model.sh CMCL/SAN/m5_k3 ensemble clevr 0 2 0

# MCL with k=1,2,3
bash scripts/train_model.sh MCL/SAN/m5_k1 ensemble clevr 0 2 0
bash scripts/train_model.sh MCL/SAN/m5_k2 ensemble clevr 0 2 0
bash scripts/train_model.sh MCL/SAN/m5_k3 ensemble clevr 0 2 0
```


### Evaluating VQA Network

You can evaluate VQA Network with following command or running `eval_model.sh` in scripts folder.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.eval \
	--exp MCL-KD/SAN/m5_k3 \
	--model_type ensemble \
	--dataset clevr \
	--start_epoch 0 \
	--end_epoch 100 \
	--epoch_stride 5 \
	--num_workers 4 \
```
