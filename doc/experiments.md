# Experiments

## CLEVR Experiments

### Training CMCL-based VQA Network

You can traing CMCL-based VQA Network with following command or train_model.sh in script folder.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.experiment.train \
	--config_path src/experiment/options/san/default.yml \
	--model_type san \
	--num_workers 8 \
```

### Visualization of Model's states.

Visualization of the model states gives valuable insite about how model works.
You can visualize the model's states using the following commands:

```bash
python -m src.experiment.clevr.visualize.end2end_neural_module_network \
--state_h5_file results/experiments/clevr/end2end_neural_module_network/default/states/final.h5
```
where *state_h5_file* is the location of state files with hdf5 format.
This script will save visualizations in the following path.
```
{directory of (state_h5_file)}/visualization/{name of (state_h5_file)}/{batch_index}.svg
```
