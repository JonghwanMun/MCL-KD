# Data Construction

## Vocabulary Construction and Encoding Question and Answer
```bash
bash scripts/preprocess_clevr.sh
bash scripts/preprocess_vqa.sh
```

## Image Feature Extraction
In this repository, only support to extract conv4 and conv5 features from ResNet-101. </br >
Running following commands or typing `bash scripts/feat_extract.sh`.
```bash
python -m src.preprocess.feat_extract \
	--save_dir data/CLEVR_v1.0/feats \
	--image_dir data/coco \
	--feat_type conv4 \
	--num_batch 128 \
	--image_size 224 \
	--data_type numpy \
```

## Options for preprocessing
### Question Alignment

Note that preprocessed question tokens are left aligned in default.
For example, left aligned question looks as follows:
```
how mani red shini cylind are there ? EMPTY EMPTY EMPTY
```

If you want to right align the preprocessed question, set --use_right_align option.
The right aligned question looks as follows:
```
Question: EMPTY EMPTY EMPTY EMPTY what number of block are there ?
```
