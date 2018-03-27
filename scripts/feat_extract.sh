CUDA_VISIBLE_DEVICES=$1 python -m src.preprocess.feat_extract \
	--save_dir data/VQA_v2.0/feats/resnet_conv4_feats \
	--image_dir data/coco \
	--feat_type conv4 \
	--num_batch 128 \
	--image_size 224 \
	--data_type numpy \
