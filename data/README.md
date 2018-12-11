# Data Directory

#### 0. Folder Hierarchy
- **CLEVR_v1.0**: Directory containing CLEVR v1.0 dataset.
- **models**: Directory containing pre-trained models.


#### 1. Download CLEVR Dataset
If you want to download data from web, use the following scripts:
```bash
./download_clevr.sh
```
After downloading data, preprocess (building vocabulary, converting questions and answers to label and extracting image features) the data with following script:
```bash
cd ..
bash scripts/preprocess_clevr.sh
bash scripts/feat_extract.sh 0
```

#### 2. Download Pre-trained Models
```bash
./download_pretrained_models.sh
```
