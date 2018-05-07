from __future__ import absolute_import ,division ,print_function

import os
import pdb
import json
import h5py
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import random; random.seed(1234)
from collections import OrderedDict

from PIL import Image
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize

import torch
import torch.utils.data as data
import torchvision.transforms as trn

from src.utils import utils, io_utils

def GetConfig(data_config_file):
    with open(data_config_file, "r") as f:
        config = yaml.load(f)
        print ("Loaded config:")
        print (json.dumps(config, indent=4))
        return config


def GetQuestionsFromFile(config, split):
    if split in config["split_options"]:
        if split in ["train", "val"]:
            questions = json.load(open(os.path.join(
                config["vqa_root"], config["question_json"] % split), "r"))
        elif split == "test":
            questions = json.load(open(os.path.join(
                config["vqa_root"], config["test_question_json"]), "r"))
        else:
            questions = json.load(open(os.path.join(
                config["vqa_root"], config["test_dev_question_json"]), "r"))
        print("The number of questions for %s: %d" % (split, len(questions["questions"])))
        return questions
    else:
        return {}


def GetAnnotationsFromFile(config, split):
    if split in ["train", "val"]:
        annotations = json.load(open(os.path.join(
            config["vqa_root"], config["annotation_json"] % split), "r"))

        for ann in annotations["annotations"]:
            ann["data_type"] = split  # save the data_type [train|val|test]
            if split in ["test-dev", "test"]:
                ann["image_filename"] = \
                    config["test_image_path"] % ("test", "test", ann["image_id"])
            else:
                ann["image_filename"] = \
                    config["image_path"] % (split, split, ann["image_id"])
        print("The number of annotations for %s: %d".format( \
                split, len(annotations["annotations"])))
        return annotations
    else:
        return {}


def GetComplementaryPairsFromFile(config, split):
    if split in ["train", "val"]:
        complementary_pair = json.load(open(os.path.join(
            config["vqa_root"], config["complementary_pair_json"] % split), "r"))
        print("The number of complementary pairs for %s: %d" % (split, len(complementary_pair)))
        return complementary_pair
    else:
        return []

""" Temporary method for efficient interactive usage.
Args:
    splits: List of target splits. The default value is ["val"] as it contains a
    minimum amount of data.
Returns:
    Instance of VQADataset class.
"""
def GetVQADataset(splits=["val"], config_path="./data/vqa_v2.0_path.yml", skip_annotation=False):
    print ("GetVQADataset: [{}]".format(", ".join(splits)))
    return VQA(splits, data_config_file=config_path, skip_annotation=skip_annotation)

"""
Class for loading VQA dataset
"""
class VQA():
    def __init__(self, splits, data_config_file="./data/vqa_v2.0_path.yml", skip_annotation=False):
        if type(splits) != list:
            raise ValueError("The argument splits should be a list of split names")

        # Get path configuration for VQA dataset.
        self.config = GetConfig(data_config_file)

        if not set(splits).issubset(set(self.config["split_options"])):
            raise ValueError("Unknown split option for current data configuration")

        # Read questions
        print ("Read questions, annotations and complementary pairs .. ")
        self.questions = []
        self.annotations = []
        self.complementary_pairs = []
        for split in splits:
            question_dict = GetQuestionsFromFile(self.config, split)
            complementary_pair_list = GetComplementaryPairsFromFile(self.config, split)
            questions = question_dict["questions"]
            for q in questions:
                if split == "test-dev" or split == "test":
                    q["image_filename"] = self.config["test_image_path"] % ("test", "test", q["image_id"])
                else:
                    q["image_filename"] = self.config["image_path"] % (split, split, q["image_id"])
            self.questions.extend(questions)
            if not skip_annotation:
                annotation_dict = GetAnnotationsFromFile(self.config, split)
                self.annotations.extend(annotation_dict["annotations"])
            else:
                self.annotations = [""] * len(self.questions)
            self.complementary_pairs.extend(complementary_pair_list)
        self.SetQuestionToAnnotationDict()
        self.splits = splits
        print ("done.")

        # Computation flags
        self.is_tokenized = False
        self.has_vocabulary = False

        # Possible vocabulary options
        # raw: use words without any processing.
        # stem: apply stemmer to words - mainly to make plural and sigular the same.
        self.vocab_options = ["raw", "stem"]

    def SetQuestionToAnnotationDict(self):
        self.qst2ann = {}
        for q,a in zip(self.questions, self.annotations):
            self.qst2ann[q["question_id"]] = a

    def LoadVocabulary(self, vocab_dict):
        """ Load pre-computed Vocabulary
        """
        self.word_vocabulary = vocab_dict["vocabulary"]
        self.word_index = vocab_dict["word_index"]
        self.has_vocabulary = True

        # Tokenize sampled questions.
        if not self.is_tokenized:
            self.TokenizeQuestions()

    def ConstructVocabulary(self, vocab_option="stem", threshold_num_answer=2000):
        """ Construct Vocabulary by performming sentence tokenization and word stemming.
        Args:
            vocab_option: how to process the vocabulary [raw|stem]
        """
        if vocab_option not in self.vocab_options:
            raise ValueError("Unknown vocabulary option is given")

        if vocab_option == "stem":
            stemmer = PorterStemmer()

        print ("Construct vocabulary (vocabulary option: {}, threshold:{})"\
                .format(vocab_option, threshold_num_answer))

        # Count vocabulary for the answers
        answer_word_frequency = OrderedDict()
        for a in tqdm(self.annotations):
            if "multiple_choice_answer" in a:
                ans = a["multiple_choice_answer"]
                if vocab_option == "stem":
                    ans = stemmer.stem(ans)
                    a["multiple_choice_answer"] = ans
                answer_word_frequency[ans] = answer_word_frequency.get(ans, 0) + 1

        # Sampling the 2K most frequent answers
        sorted_answer_word_frequency = OrderedDict(sorted(answer_word_frequency.items(), \
                                                          key=lambda x:x[1], reverse=True))
        answer_word_frequency = OrderedDict()
        bad_answer_word_frequency = OrderedDict()
        print (list(sorted_answer_word_frequency.keys())[:30])
        for ia, k in enumerate(sorted_answer_word_frequency.keys()):
            if ia < threshold_num_answer:
                answer_word_frequency[k] = sorted_answer_word_frequency[k]
            else:
                bad_answer_word_frequency[k] = sorted_answer_word_frequency[k]
                #self.PrintAnswerVocabularyCoverage(answer_word_frequency)

        # Sampling questions and annotations where the answer exists in word vocabulary
        self.SamplingQuestionsAnnotationsWithFrequentAnswers(list(answer_word_frequency.keys()))

        # Tokenize sampled questions.
        if not self.is_tokenized:
            self.TokenizeQuestions()

        # Count vocabulary for the tokenized questions.
        question_word_frequency = {}
        for q in tqdm(self.questions):
            if "vocab_option" not in q:
                q["vocab_option"] = vocab_option
            for i, w in enumerate(q["tokenized_question"]):
                if vocab_option == "stem":
                    w = stemmer.stem(w)
                    q["tokenized_question"][i] = w
                question_word_frequency[w] = question_word_frequency.get(w, 0) + 1

        # Merge word counts for question and answer words
        self.word_frequency = {"total":{}, "question":{}, "answer": {}}
        for w in set(list(question_word_frequency.keys()) + list(answer_word_frequency.keys())):
            self.word_frequency["total"][w] = \
                question_word_frequency.get(w, 0) + answer_word_frequency.get(w, 0)
        self.word_frequency["question"] = question_word_frequency
        self.word_frequency["answer"] = answer_word_frequency

        # Construct word vocabulary for question and answer
        self.word_vocabulary = {}
        self.word_vocabulary["total"] = sorted(self.word_frequency["total"].keys())
        self.word_vocabulary["question"] = sorted(self.word_frequency["question"].keys())
        self.word_vocabulary["answer"] = sorted(self.word_frequency["answer"].keys())

        # Construct word_index table for question, answer and total
        self.word_index = {"total": {}, "question": {}, "answer": {}}
        self.word_index["total"] = \
            {w: i+2 for i, w in enumerate(self.word_vocabulary["total"])}
        self.word_index["question"] = \
            {w: i+2 for i, w in enumerate(self.word_vocabulary["question"])}
        self.word_index["answer"] = \
            {w: i for i, w in enumerate(self.word_vocabulary["answer"])}
        # Set EMPTY and UNKNOWN words at first and second location in vocab
        for k in ["total", "question"]:
            self.word_index[k]["EMPTY"] = 0
            self.word_index[k]["UNKNOWN"] = 1
        self.has_vocabulary = True

        # Print vocabulary sizes of questions and answer
        print("The size of question vocabulary: {}".format( \
                len(self.word_vocabulary["question"])))
        print("The size of answer vocabulary: {}".format( \
                len(self.word_vocabulary["answer"])))
        print("The size of total vocabulary: {}".format( \
                len(self.word_vocabulary["total"])))

    def SamplingQuestionsAnnotationsWithFrequentAnswers(self, answer_vocabulary=None):
        if answer_vocabulary == None :
            answer_vocabulary = self.word_vocabulary["answer"]
        tmp_questions = []
        tmp_annotations = []

        for q,a in tqdm(zip(self.questions, self.annotations)):
            if a["multiple_choice_answer"] in answer_vocabulary:
                tmp_questions.append(q)
                tmp_annotations.append(a)

        # check whether the order of questions and annotations is same
        for q,a in zip(tmp_questions, tmp_annotations):
            if q["question_id"] != a["question_id"]:
                raise ValueError("Incorrect order of questions and annotations")

        print ("The number of questions/annotations before sampling {}/{}".format( \
                len(self.questions), len(self.annotations)))
        self.questions = tmp_questions
        self.annotations = tmp_annotations
        print ("The number of questions/annotations after sampling {}/{}".format( \
                len(self.questions), len(self.annotations)))

    def PrintAnswerVocabularyCoverage(self, answer_vocabulary=None):
        """
        This function should be called after self.ConstructVocabulary() or
        self.LoadVocabulary()
        """
        if answer_vocabulary == None :
            answer_vocabulary = self.word_vocabulary["answer"]

        num_ans = 0
        total_anns = len(self.annotations)
        for ann in tqdm(self.annotations):
            ans = ann["multiple_choice_answer"]
            if ans in answer_vocabulary:
                num_ans += 1

        print("Answer vocabulary of size {} covers {:.3f}% of questions".format( \
                len(ans_vocabulary), num_ans*100.0/total_anns))

    def GetWordVocabulary(self):
        if not self.has_vocabulary:
            self.ConstructVocabulary()
        return self.word_vocabulary

    def GetWordIndexes(self):
        if not self.has_vocabulary:
            self.ConstructVocabulary()
        return self.word_index

    def GetWordFrequency(self):
        if not self.has_vocabulary:
            self.ConstructVocabulary()
        return self.word_frequency

    def GetImagePathsByQuestions(self, questions):
        return [self.GetImagePathByQuestion(q) for q in questions]

    def GetImagePathByQuestion(self, question):
        ann = self.qst2ann[question["question_id"]]
        image_path = os.path.join(self.config["vqa_root"],
                                  "images/{}".format(ann["image_filename"]))
        return image_path

    def GetShortQuestions(self, max_question_length=0):
        if not self.is_tokenized:
            self.TokenizeQuestions()

        short_questions = []
        short_annotations = []
        for question, annotation in tqdm(zip(self.questions, self.annotations)):
            if question["question_length"] <= max_question_length:
                short_questions.append(question)
                short_annotations.append(annotation)
        return short_questions, short_annotations

    def GetQuestionsByIds(self, question_ids, tokenize=True):
        if (not self.is_tokenized) and tokenize:
            self.TokenizeQuestions()

        qid_to_question = {q["question_id"]: q
                           for q,a in zip(self.questions, self.annotations)}
        questions = [qid_to_question[qid] for qid in question_ids]
        annotations = [self.qst2ann[q["question_id"]] for q in questions]

        return questions, annotations

    def GetQuestions(self):
        if not self.is_tokenized:
            self.TokenizeQuestions()
        return self.questions

    def GetAnnotations(self):
        return self.annotations

    def GetQuestionsAndAnnotations(self):
        if not self.is_tokenized:
            self.TokenizeQuestions()
        return self.questions, self.annotations

    def GetMaxQuestionLength(self):
        if not self.is_tokenized:
            self.TokenizeQuestions()
        return max([q["question_length"] for q in self.questions])

    def GetQuestionsWithLength(self, question_length):
        if not self.is_tokenized:
            self.TokenizeQuestions()

        target_questions = []
        target_annotations = []
        for question, annotation in tqdm(zip(self.questions, self.annotaions)):
            if question["question_length"] == question_length:
                target_questions.append(question)
                target_annotations.append(annotation)
        return target_questions, target_annotations

    def GetQuestionStatistics(self):
        if not self.is_tokenized:
            self.TokenizeQuestions()

        stats = {}
        stats["max_question_length"] = max([q["question_length"] for q in self.questions])
        stats["number_of_questions_per_length"] =  []
        stats["number_of_unique_questions_per_length"] = []
        for i in tqdm(range(stats["max_question_length"])):
            qi = [q["question"] for q in self.GetQuestionsWithLength(i+1)]
            stats["number_of_questions_per_length"].append(len(qi))
            stats["number_of_unique_questions_per_length"].append(len(list(set(qi))))
        return stats


    """ Tokenize questions.
    Tokenize questions ans save them and their length in questions.
    """
    def TokenizeQuestions(self):
        print ("Tokenize questions")
        for question in tqdm(self.questions):
            question["tokenized_question"] = word_tokenize(str(question["question"]).lower())
            question["question_length"] = len(question["tokenized_question"])
            self.is_tokenized = True


"""
Wrapper class of loader (pytorch provided) for VQA v2.0.
"""
class DataSet(data.Dataset):
    def __init__(self, config):

        # get configurations
        print(json.dumps(config, indent=4))
        self.hdf5_path = utils.get_value_from_dict(
            config, "encoded_hdf5_path", \
            "data/VQA_v2.0/preprocess/encoded_qa/vqa_2000_vocab_train-val_raw/" \
            + "all_questions_with_answer_vocab_use_zero_token_max_qst_len_23/" \
            + "qa_train.h5")
        self.json_path = utils.get_value_from_dict(
            config, "encoded_json_path", \
            "data/VQA_v2.0/preprocess/encoded_qa/vqa_2000_vocab_train-val_raw/" \
            + "all_questions_with_answer_vocab_use_zero_token_max_qst_len_23/" \
            + "qa_train.json")
        self.img_size = utils.get_value_from_dict(config, "img_size", 224)
        self.batch_size = utils.get_value_from_dict(config, "batch_size", 64)
        self.use_gpu = utils.get_value_from_dict(config, "use_gpu", True)
        self.use_img = utils.get_value_from_dict(config, "use_img", False)
        if self.use_img:
            self.img_dir = utils.get_value_from_dict(
                config, "img_dir", "data/VQA_v2.0/images")
            self.prepro = trn.Compose([
                trn.Resize(self.img_size),
                trn.CenterCrop(self.img_size),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.feat_type = utils.get_value_from_dict(config, "feat_type", "numpy")
            if self.feat_type == "numpy":
                self.feat_dir = utils.get_value_from_dict(
                    config, "feat_dir", "data/VQA_v2.0/feats/tmp")
            else:
                self.feat_path = utils.get_value_from_dict(
                    config, "feat_path", "data/VQA_v2.0/feats/resnet_conv4_feats/img_feats.h5")
        self.fetching_answer_option = utils.get_value_from_dict(
                config, "fetching_answer_option", "sampling")
        self.only_question = utils.get_value_from_dict(config, "only_question", False)

        # load hdf5 file including question_labels, question_length,
        # answer_labels
        hdf5_file = io_utils.load_hdf5(self.hdf5_path)
        self.max_time_steps = hdf5_file["question_labels"].shape[1]

        # load json file including woti, itow, atoi, itoa, splits, vocab_info,
        # question_ids, image_filenames
        self.json_file = io_utils.load_json(self.json_path)
        if self.fetching_answer_option == "sampling":
            self.second_frequent_answer_labels = self.json_file["second_frequent_answer_labels"]

        # set path of pre-computed assignments
        # NOTE: DEPRECATED
        self.assignment_path = utils.get_value_from_dict(config, "assignment_path", "")

        # set path of pre-computed logits of base models
        self.base_logits_path = utils.get_value_from_dict(config, "base_logits_path", "")

    def __getitem__(self, idx):
        """ Retrun a data (images, question_label, question length and answers)
        Returns:
            img (or feat): image (or feature)
            qst_label: question label
            qst_length: question length
            answer: answer for questions
        """

        # obtain image (in raw or feature)
        img_filename = self.json_file["image_filenames"][idx]
        if self.only_question:
            img = torch.zeros(1)
        else:
            if self.use_img:
                img_path = os.path.join(self.img_dir, img_filename)
                img = Image.open(img_path).convert("RGB")
                img = self.prepro(img)

            else:
                if self.feat_type == "numpy":
                    feat_path = os.path.join(self.feat_dir, \
                            img_filename.replace(".jpg", ".npy"))
                    img = np.load(feat_path)
                else:
                    feat_hdf5 = io_utils.load_hdf5(self.feat_path, verbose=False)
                    feat_key = img_filename.split("/")[-1][:-4]
                    img = feat_hdf5[feat_key]
                img = torch.Tensor(img)

        # obtain question label and its length
        hdf5_file = io_utils.load_hdf5(self.hdf5_path, verbose=False)
        qst_label = torch.from_numpy(hdf5_file["question_labels"][idx])
        qst_length = hdf5_file["question_length"][idx]

        # obtain answer label
        if self.fetching_answer_option == "all_answers":
            answer = [hdf5_file["most_frequent_answer_labels"][idx],
                      hdf5_file["all_answer_labels"][idx],
                      hdf5_file["all_answer_labels_mask"][idx]]
            answer = [torch.from_numpy(np.asarray([ans])).long() for ans in answer]

        elif self.fetching_answer_option == "sampling":
            prob = torch.rand(1)[0]
            if prob >= self.second_frequent_answer_labels[idx]["p"]:
                answer = hdf5_file["most_frequent_answer_labels"][idx]
            else:
                answer = np.array(self.second_frequent_answer_labels[idx]["answer"])
            answer = torch.from_numpy(np.asarray([answer])).long()

        elif self.fetching_answer_option == "simple":
            answer = hdf5_file["most_frequent_answer_labels"][idx]
            answer = torch.from_numpy(np.asarray([answer])).long()

        elif self.fetching_answer_option == "without_answer":
            # we use garbage answer (of 1)
            answer = torch.from_numpy(np.asarray([1])).long()
        else:
            raise ValueError("Not supported fetching answer option ({})".format(
                self.fetching_answer_option))

        # obtain img info (question id)
        qst_id = self.json_file["question_ids"][idx]

        # prepare batch output
        out = [img, qst_label, qst_length]
        if self.assignment_path != "":
            # NOTE: DEPRECATED
            # obtain assignment label
            assignment_file = io_utils.load_hdf5(self.assignment_path, verbose=False)
            assignments = torch.from_numpy(assignment_file["assignments"][idx])
            out.append(assignments)
        if self.base_logits_path != "":
            base_logits = io_utils.load_hdf5(self.base_logits_path, verbose=False)
            base_logits = torch.from_numpy(base_logits["base_logits"][idx])
            out.append(base_logits)
        out.append(answer)
        out.append(qst_id)
        return out

    def __len__(self):
        return len(self.json_file["image_filenames"])

    def get_samples(self, num_samples):
        """ Retrun sample data (images, question_label question_legnth, answers, filename)
        Returns:
            img (or feat): image (or feature)
            qst_label: question label
            qst_length: question length
            answer: answer for questions
            img_filename: filename of images
        """

        samples = []
        cur_num_samples = 0
        sample_answers = []
        while True:
            # randomly select index of sample
            idx = np.random.randint(0, len(self)-1)

            # obtain question label, its length and answer
            img_filename = self.json_file["image_filenames"][idx]
            sample = self.__getitem__(idx)

            # TODO
            if self.fetching_answer_option == "without_answer":
                samples.append([*sample[:-1], img_filename])
                cur_num_samples += 1
            else:
                cur_answer_label = sample[-2][0][0]
                random_prob = np.random.rand(1)[0]
                if random_prob < 0.001:
                    sample_answers.append(cur_answer_label)
                    samples.append([*sample[:-1], img_filename])
                    cur_num_samples += 1
                else:
                    # we get samples with different answers
                    if cur_answer_label in sample_answers:
                        continue
                    else:
                        sample_answers.append(cur_answer_label)
                        samples.append([*sample[:-1], img_filename])
                        cur_num_samples += 1

            if cur_num_samples == num_samples:
                break

        return collate_fn(samples)

    def get_iter_per_epoch(self):
        return len(self) / self.batch_size

    def get_max_time_steps(self):
        return self.max_time_steps

    def get_idx_empty_word(self):
        return self.json_file["wtoi"]["EMPTY"]

    def get_qst_ids(self):
        return self.json_file["question_ids"]

    def get_vocab_size(self):
        return len(self.json_file["wtoi"])

    def get_num_answers(self):
        return len(self.json_file["atoi"])

    def get_wtoi(self):
        return self.json_file["wtoi"]

    def get_itow(self):
        return self.json_file["itow"]

    def get_atoi(self):
        return self.json_file["atoi"]

    def get_itoa(self):
        return self.json_file["itoa"]

def collate_fn(batch):
    """Creates mini-batch tensors from the list of items
        (image, question_label, question_length, answer).

    We should build custom collate_fn rather than using default collate_fn,
    because merging question (including padding) is not supported in default.
    Args:
        batch: list of items (image, question_label, question_length, answer).
            - image: torch tensor of image or feature [?].
            - question_label: torch tensor [padded_length]; variable length.
            - question_lengths: scalar.
            - answers: scalar.
    Returns:
        imgs: torch FloatTensor [batch_size, feat_dim, kernel_size, kernel_size].
        qst_labels: torch LongTensor [batch_size, padded_length].
        qst_lengths: list whose length is batch_size; valid length for each padded caption.
        answers: torch LongTensor [batch_size].
    """
    # sort a batch by question length (descending order)
    batch.sort(key=lambda x: x[2], reverse=True)
    batch = [val for val in zip(*batch)]
    """ imgs, qst_labels, qst_lengths, answers = zip(*batch) """

    items = batch[:-1]
    img_info = batch[-1] # qst_id or img_filename
    return_vals = [items, img_info]

    # stacking tensors
    num_items = len(items)
    items[0] = torch.stack(items[0], 0) # images
    items[1] = torch.stack(items[1], 0).long() # question labels
    if type(items[-1][0]) == type(list()):
        answer, all_answer, mask = zip(*items[-1])
        items[-1] = [torch.stack(answer, 0).squeeze(),       # most frequent answers
                     torch.stack(all_answer, 0).squeeze(),   # all answers
                     torch.stack(mask, 0).float().squeeze()] # mask for all answers
    else:
        items[-1] = torch.stack(items[-1], 0).squeeze() # from scalar (tuple) to 1D tensor
    if num_items == 5:
        items[-2] = torch.stack(items[-2], 0) # for pre-computed logits of base models
    return return_vals


