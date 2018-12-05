CUDA_VISIBLE_DEVICES=$1 python -m src.preprocess.feat_extract \
	--save_dir data/VQA_v2.0/feats/resnet_conv4_feats/test2015/ \
	--image_dir /data/jonghwan/VQA_v2.0/images/test2015/ \
	--feat_type conv4 \
	--num_batch 128 \
	--image_size 224 \
	--data_type numpy \
	--debug_mode
