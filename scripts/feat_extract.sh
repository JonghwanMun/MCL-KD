## Extract features for training set
CUDA_VISIBLE_DEVICES=$1 python -m src.preprocess.feat_extract \
	--save_dir data/CLEVR_v1.0/feats/resnet_conv4_feats/train/ \
	--image_dir data/CLEVR_v1.0/images/train/ \
	--feat_type conv4 \
	--num_batch 128 \
	--image_size 224 \
	--data_type numpy \
	--debug_mode

## Extract features for validation set
CUDA_VISIBLE_DEVICES=$1 python -m src.preprocess.feat_extract \
	--save_dir data/CLEVR_v1.0/feats/resnet_conv4_feats/val/ \
	--image_dir data/CLEVR_v1.0/images/val/ \
	--feat_type conv4 \
	--num_batch 128 \
	--image_size 224 \
	--data_type numpy \
	--debug_mode
