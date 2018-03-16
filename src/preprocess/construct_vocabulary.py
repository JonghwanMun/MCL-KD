import os
import json
import argparse

from src.dataset import clevr_dataset
from src.dataset import vqa_dataset

def main():
    parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vocab_option", default="raw",
            help="Options used for preprocessing vocabulary [raw, stem]")
    parser.add_argument("--target_splits", nargs='+', default=['train', 'val'],
            help="List of target splits. Subset of 'train', 'val', 'test' can be used")
    parser.add_argument("--save_vocab_dir", default="data/CLEVR_v1.0/preprocess/vocabulary",
            help="Path for saving constructed vocabulary")
    parser.add_argument("--dataset", default="clevr",
            help="Dataset [clevr, vqa]")
    parser.add_argument("--dataset_config_path", default="./data/clevr_path.yml",
            help="Path to dataset config file")
    parser.add_argument("--threshold_num_answer", default=2000,
            help="The number of the most frequent answers")
    args = parser.parse_args()
    print (args)

    print ("Construct vocabulary for the %s dataset" % args.dataset)
    if not os.path.isdir(args.save_vocab_dir):
        print ("Directory doesn't exist, create one: {}".format(args.save_vocab_dir))
        os.makedirs(args.save_vocab_dir)

    # Construct vocabulary
    if args.dataset == 'clevr':
        dts = clevr_dataset.GetClevrDataset(splits=args.target_splits,
            clevr_config=args.dataset_config_path)
        dts.ConstructVocabulary(vocab_option=args.vocab_option)
    elif args.dataset == 'vqa':
        dts = vqa_dataset.GetVQADataset(args.target_splits, args.dataset_config_path)
        dts.ConstructVocabulary(vocab_option=args.vocab_option,
                threshold_num_answer=int(args.threshold_num_answer))
    else:
        raise ValueError('The argument dataset should be one among [clevr, vqa]')

    # Organize vocabulary info in dictionary
    vocabulary_set = {}
    vocabulary_set['vocabulary'] = dts.GetWordVocabulary()
    vocabulary_set['word_index'] = dts.GetWordIndexes()
    vocabulary_set['info'] = {'splits':args.target_splits,
                'vocab_option':args.vocab_option}
    if args.dataset == 'vqa':
        vocabulary_set['info']['threshold_num_answer'] = args.threshold_num_answer

    # Save vocabulary info as a json file
    if args.dataset == 'clevr':
        save_vocab_filename = "{}_vocab_{}_{}.json".format(args.dataset, \
                '-'.join(sorted(args.target_splits)), args.vocab_option)
    elif args.dataset == 'vqa':
        save_vocab_filename = "{}_{}_vocab_{}_{}.json".format(args.dataset, \
                args.threshold_num_answer, '-'.join(sorted( \
                args.target_splits)), args.vocab_option)
    save_vocab_path = os.path.join(args.save_vocab_dir, save_vocab_filename)
    print ("Save vocabulary in : {}".format(save_vocab_path))
    json.dump(vocabulary_set, open(save_vocab_path, 'w'), indent=4)


if __name__ == "__main__":
    main()
