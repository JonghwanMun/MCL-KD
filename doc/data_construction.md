# Data Construction

## Vocabulary Construction.

### In ipython environment

You can construct vocabulary with the following commands:

```python
from src.dataset import clevr_dataset
clv = clevr_dataset.GetClevrDataset(['train'])
clv.ConstructVocabulary(vocab_option='stem')
```
Note that vocab\_option chooses whether to use stemming or not.
If you don't want to use stemming, use option: ```vocab_option='raw'```
Constructed vocabulary is saved in the clevr dataset class instance.

### In commandline

You can construct vocabulary json file with following commands:

```bash
python -m src.preprocess.construct_vocabulary \
  --vocab_option 'stem' --target_splits 'train' 'val' \
  --save_vocab_dir data/preprocess/vocabulary
```
This command will save vocabulary file in the following path:
```
data/preprocess/vocabulary/vocab_train-val_stem.json
```

## Encode Question and Answer

To use questions and answers for training, they should be converted to the
arrays of integers. The following script encode questions and answers based
on pre-computed vocabulary.

```bash
python -m src.preprocess.preprocess_clevr \
  --target_splits 'train' 'val' \
  --vocab_path 'data/preprocess/vocabulary/vocab_train-val_stem.json' \
  --save_encoded_qa_dir 'data/preprocess/encoded_qa'
```

#### Question Alignment

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

#### Zero Tokens

Sometimes lua makes error if zero tokens are given to some layers.
[LookupTableMaskZero](https://github.com/Element-Research/rnn#rnn.LookupTableMaskZero)
module handles this problem by return zero vector for zero tokens, but you should
be careful of using zero tokens in general cases.

Preprocessing script fills empty tokens with ones by default, but you can also fill the
empty tokens with zeros by setting the flag --use_zero_token

## Visualization of Encoded Question and Answer

Visualizing encoded question and answer is required for debugging; it could be
used for checking whether the questions and answers are correctly encoded.
You can use the following code for visualization.

```bash
python -m src.analysis.visualize_encoded_qa
```
