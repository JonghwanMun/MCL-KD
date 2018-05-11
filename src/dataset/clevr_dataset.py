from __future__ import absolute_import ,division ,print_function

import os
import pdb
import json
import h5py
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as trn

import yaml
from tqdm import tqdm
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize

from src.utils import utils, io_utils

CLEVR_CONFIG="./data/clevr_path.yml"

def GetConfig(data_config_file):
	with open(data_config_file, "r") as f:
		config = yaml.load(f)
		print ("Loaded config:")
		print (json.dumps(config, indent=4))
	return config


def GetQuestionsFromFile(config, split):
	if split in config["split_options"]:
		questions = json.load(open(os.path.join(
			config["clevr_root"], config["question_json"] % split), "r"))
		return questions
	else:
		return {}

def GetSceneFromFile(config, split):
    if split in config["scene_split_options"]:
        scene_dict = json.load(open(os.path.join(
            config["clevr_root"], config["scene_json"] % split), "r"))
        return scene_dict
    else:
        return {}

"""Temporary method for efficient interactive usage.
Args:
	splits: List of target splits. The default value is ["val"] as it contains a
		minimum amount of data.
Returns:
	Instance of Clevr class.
"""
def GetClevrDataset(splits=["val"], clevr_config="./data/clevr_path.yml",
                    load_scene=False, verbose=False):
	print ("GetClevr: [{}]".format(", ".join(splits)))
	return Clevr(clevr_config, splits, load_scene, verbose)


class Clevr():
    """Initialize Clevr class.
    """
    def __init__(self, data_config_file, splits, load_scene=False, verbose=False):
        if type(splits) != list:
            raise ValueError("The argument splits should be a list of split names")

        # Get path configuration for CLEVR dataset.
        # TODO: There is a configuration missmatches in split variable. Need to fix
        self.config = GetConfig(data_config_file)

        if not set(splits).issubset(set(self.config["split_options"])):
            raise ValueError("Unknown split option for current data configuration")


        # Read questions
        print ("Read questions.. ", end="")
        self.questions_info = []
        self.questions = []
        for split in splits:
            question_dict = GetQuestionsFromFile(self.config, split)
            self.questions_info.append(question_dict["info"])
            self.questions.extend(question_dict["questions"])
        print ("done.")

        # Read scenes
        if load_scene:
            print ("Read scenes.. ")
            self.scene_info = []
            self.scenes = []
            for split in splits:
                scene_dict = GetSceneFromFile(self.config, split)
                self.scene_info.append(scene_dict["info"])
                self.scenes.extend(scene_dict["scenes"])
            self.scene_dict = {scene["image_filename"] : scene
                         for scene in self.scenes}
            print ("done.")

        self.splits = splits
        self.load_scene = load_scene
        self.verbose = verbose

        # Computation flags
        self.is_tokenized = False
        self.has_vocabulary = False

        # Possible vocabulary options
        # raw: use words without any processing.
        # stem: apply stemmer to words - mainly to make plural and sigular the same.
        self.vocab_options = ["raw", "stem"]

    """ Load Vocabulary

    Args:
        vocab_path: path to the pre-computed vocabulary
    """
    def LoadVocabulary(self, vocab_path):

        vocab_dict = io_utils.load_json(vocab_path)
        self.word_vocabulary = vocab_dict["vocabulary"]
        self.word_index = vocab_dict["word_index"]
        self.has_vocabulary = True

        """ TODO: should do this part?
        # Tokenize sampled questions.
        if not self.is_tokenized:
            self.TokenizeQuestions()
        """

    """ Construct Vocabulary

    Construct vocabulary by performming sentence tokenization (and word stemming).
    Args:
        vocab_option: how to process the vocabulary [raw|stem]
    """
    def ConstructVocabulary(self, vocab_option="raw"):
        if vocab_option not in self.vocab_options:
            raise ValueError("Unknown vocabulary option is given")

        if not self.is_tokenized:
            self.TokenizeQuestions()

        if vocab_option == "stem":
            stemmer = PorterStemmer()

        print ("Construct vocabulary (vocabulary option: {})".format(vocab_option))
        question_word_frequency = {}
        answer_word_frequency = {}
        self.vocab_token_dict = {}
        for q in tqdm(self.questions):
            if "vocab_option" not in q:
                q["vocab_option"] = vocab_option
            # Count vocabulary for the tokenized questions.
            for i, w in enumerate(q["tokenized_question"]):
                if vocab_option == "stem":
                    if w not in self.vocab_token_dict.keys():
                        self.vocab_token_dict[w] = stemmer.stem(w)
                    w = self.vocab_token_dict[w]
                    q["tokenized_question"][i] = w
                else:
                    if w not in self.vocab_token_dict.keys():
                        self.vocab_token_dict[w] = w
                question_word_frequency[w] = \
                        question_word_frequency.get(w, 0) + 1
			# Count vocabulary for the answers
            if "answer" in q:
                ans = q["answer"]
                if vocab_option == "stem":
                    if ans not in self.vocab_token_dict.keys():
                        self.vocab_token_dict[ans] = stemmer.stem(ans)
                    ans = self.vocab_token_dict[ans]
                    q["answer"] = ans
                else:
                    if ans not in self.vocab_token_dict.keys():
                        self.vocab_token_dict[ans] = ans
                answer_word_frequency[ans] = \
                    answer_word_frequency.get(ans, 0) + 1

        # TODO: consider when "vocab_option" is raw
        if self.load_scene and (vocab_option == "stem"):
            for scene in self.scenes:
                for obj in scene["objects"]:
                    for attr_key in ["color", "material", "shape", "size"]:
                        obj[attr_key] = self.vocab_token_dict[obj[attr_key]]

        # Merge word counts for question and answer words
        self.word_frequency = {"total":{}, "question":{}, "answer": {}}
        for w in set(list(question_word_frequency.keys()) + list(answer_word_frequency.keys())):
            self.word_frequency["total"][w] = \
                question_word_frequency.get(w, 0) + answer_word_frequency.get(w, 0)

        self.word_frequency["question"] = question_word_frequency
        self.word_frequency["answer"] = answer_word_frequency

        # Construct word vocabulary
        self.word_vocabulary = {}
        self.word_vocabulary["total"] = sorted(self.word_frequency["total"].keys())
        self.word_vocabulary["question"] = sorted(self.word_frequency["question"].keys())
        self.word_vocabulary["answer"] = sorted(self.word_frequency["answer"].keys())

        # Construct word_index table for question, answer and total
        # TODO: check start index is 0 or 1?
        self.word_index = {"total": {}, "question": {}, "answer": {}}
        self.word_index["total"] = \
            {w: i+2 for i, w in enumerate(self.word_vocabulary["total"])}
        self.word_index["question"] = \
            {w: i+2 for i, w in enumerate(self.word_vocabulary["question"])}
        self.word_index["answer"] = \
            {w: i for i, w in enumerate(self.word_vocabulary["answer"])}
        # Set EMPTY and UNKNOWN words at first and second location in vocab
        for key in ["total", "question"]:
            self.word_index[key]["EMPTY"] = 0
            self.word_index[key]["UNKNOWN"] = 1
        self.has_vocabulary = True

        # Print vocabulary sizes of questions and answer
        print("The size of question vocabulary: {}".format( \
                len(self.word_vocabulary["question"])))
        print("The size of answer vocabulary: {}".format( \
                len(self.word_vocabulary["answer"])))
        print("The size of total vocabulary: {}".format( \
                len(self.word_vocabulary["total"])))

    def GetWordVocabulary(self):
        if not self.has_vocabulary:
            self.ConstructVocabulary()
        return self.word_vocabulary

    def GetWordIndexes(self):
        if not self.has_vocabulary:
            self.ConstructVocabulary()
        return self.word_index

    def GetShortQuestions(self, max_question_length=0):
        if not self.is_tokenized:
            self.TokenizeQuestions()

        short_questions = []
        for question in tqdm(self.questions):
            if question["question_length"] <= max_question_length:
                short_questions.append(question)
        return short_questions

    def GetImagePathsByQuestions(self, questions):
        return [self.GetImagePathByQuestion(q) for q in questions]

    def GetImagePathByQuestion(self, question):
        image_path = os.path.join(self.config["clevr_root"],
                "images/{}/{}".format(question["split"], question["image_filename"]))
        return image_path

    def GetScenesByQuestions(self, questions,
                             use_vocabulary=True, use_entity=True):
        if use_vocabulary and (not self.has_vocabulary):
            self.ConstructVocabulary()

        scenes = [self.scene_dict[q["image_filename"]]
                  for q in questions]
        if use_entity:
            for scene in scenes:
                scene["color"] = [obj["color"] for obj in scene["objects"]]
                scene["size"] = [obj["size"] for obj in scene["objects"]]
                scene["material"] = [obj["material"] for obj in scene["objects"]]
                scene["shape"] = [obj["shape"] for obj in scene["objects"]]
                scene["entity"] = scene["color"] + scene["size"] + \
                    scene["material"] + scene["shape"]
        return scenes

    def GetQuestionsByIds(self, question_ids, tokenize=True):
        if (not self.is_tokenized) and tokenize:
            self.TokenizeQuestions()

        qid_to_question = {"{}_{}".format(q["split"], q["question_index"]): q
                           for q in self.questions}
        questions = [qid_to_question[qid] for qid in question_ids]

        return questions

    def GetQuestions(self, tokenize=True):
        if (not self.is_tokenized) and tokenize:
            self.TokenizeQuestions()

        return self.questions

    def GetMaxQuestionLength(self):
        if not self.is_tokenized:
            self.TokenizeQuestions()

        return max([q["question_length"] for q in self.questions])

    def GetQuestionsWithLength(self, question_length):
        if not self.is_tokenized:
            self.TokenizeQuestions()

        target_questions = []
        for question in tqdm(self.questions):
            if question["question_length"] == question_length:
                target_questions.append(question)

        return target_questions

    """Get questions with filter.

    Return questions passed the filter function. Filter functions test the
    condition of each question information and return boolean values whether
    the condition is passed or failed.

    Example:
        GetQuestionsWithFilter(question_length_filter=lambda x: x <= 8,
        question_family_index_filter=lambda x: x in [87, 88, 89])

    """
    def GetQuestionsWithFilter(self, question_length_filter=None,
                               question_family_index_filter=None):
        if not self.is_tokenized:
            self.TokenizeQuestions()

        target_questions = []
        for question in tqdm(self.questions):
            if question_length_filter != None and\
                    not question_length_filter(question["question_length"]):
                continue
            elif question_family_index_filter != None and\
                not question_family_index_filter(question["question_family_index"]):
                continue
            target_questions.append(question)
        return target_questions

    def GetQuestionStatistics(self):
        if not self.is_tokenized:
            self.TokenizeQuestions()

        stats = {}
        stats["max_question_length"] = self.GetMaxQuestionLength()
        stats["number_of_questions_per_length"] =  []
        stats["number_of_unique_questions_per_length"] = []
        for i in tqdm(range(stats["max_question_length"])):
            qi = [q["question"] for q in self.GetQuestionsWithLength(i+1)]
            stats["number_of_questions_per_length"].append(len(qi))
            stats["number_of_unique_questions_per_length"].append(len(list(set(qi))))

        return stats

    def GetSameFamilyQuestions(self, question_family_index):
        """
        Return a list of questions with same family index.

        Parameters
        ----------
        question_family_index : int, from 0 to 89
        """
        if not self.is_tokenized:
            self.TokenizeQuestions()

        if not question_family_index in range(90):
            return []

        same_question_family = [ question for question in tqdm(self.questions) \
                                if question["question_family_index"] == question_family_index]

        return same_question_family

    """Tokenize questions.

    Tokenize questions ans save them in questions.
    """
    def TokenizeQuestions(self):
        print ("Tokenize questions")
        for question in tqdm(self.questions):
            self.TokenizeSingleQuestion(question)
        self.is_tokenized = True

    def TokenizeSingleQuestion(self, question):
        question["tokenized_question"] = word_tokenize(
            str(question["question"]).lower())
        question["question_length"] = len(question["tokenized_question"])

        if self.verbose:
            print(question["tokenized_question"], question["question_length"])


"""
Wrapper class of loader (pytorch provided) for CLEVR v1.0.
"""
class DataSet(data.Dataset):

    def __init__(self, config):

        # get configions
        print(json.dumps(config, indent=4))
        self.hdf5_path = utils.get_value_from_dict(config, "encoded_hdf5_path", \
                "data/CLEVR_v1.0/preprocess/encoded_qa/vocab_train_raw/" \
                + "all_questions_use_zero_token/qa_train.h5")
        self.json_path = utils.get_value_from_dict(config, "encoded_json_path", \
                "data/CLEVR_v1.0/preprocess/encoded_qa/vocab_train_raw/" \
                + "all_questions_use_zero_token/qa_train.json")
        self.img_size = utils.get_value_from_dict(config, "img_size", 224)
        self.batch_size = utils.get_value_from_dict(config, "batch_size", 32)
        self.use_img = utils.get_value_from_dict(config, "use_img", False)
        self.use_gpu = utils.get_value_from_dict(config, "use_gpu", True)
        if self.use_img:
            self.img_dir = utils.get_value_from_dict(config, "img_dir", "data/CLEVR_v1.0/images")
            self.prepro = trn.Compose([
                trn.Resize(self.img_size),
                trn.CenterCrop(self.img_size),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.feat_dir = utils.get_value_from_dict(config, "feat_dir", "data/CLEVR_v1.0/feats")

        # load hdf5 file including question_labels, question_length,
        # answer_labels
        hdf5_file = io_utils.load_hdf5(self.hdf5_path)
        self.max_time_steps = hdf5_file["question_labels"].shape[1]

        # load json file including woti, itow, atoi, itoa, splits, vocab_info,
        # question_ids, image_filenames
        self.json_file = io_utils.load_json(self.json_path)

        # set path of pre-computed assignments
        # NOTE: DEPRECATED
        self.assignment_path = utils.get_value_from_dict(config, "assignment_path", "")

        # set path of pre-computed logits of base models
        self.base_logits_path = utils.get_value_from_dict(config, "base_logits_path", "")

        self.fetching_answer_option = "simple"

    def __getitem__(self, idx):
        """ Retrun a data (images, question_label, question length and answers)
        Returns:
            img (or feat): image (or feature)
            qst_label: question label
            qst_length: question length
            answer: answer for questions
        """

        # obtain image (as raw or feature)
        img_filename = self.json_file["image_filenames"][idx]
        if self.use_img:
            img_path = os.path.join(self.img_dir, img_filename)
            img = Image.open(img_path).convert("RGB")
            img = self.prepro(img)

        else:
            feat_path = os.path.join(self.feat_dir, img_filename.replace(".png", ".npy"))
            img = np.load(feat_path)
            img = torch.Tensor(img)

        # obtain question label and its length
        hdf5_file = io_utils.load_hdf5(self.hdf5_path, verbose=False)
        qst_label = torch.from_numpy(hdf5_file["question_labels"][idx])
        qst_length = hdf5_file["question_length"][idx]

        # obtain answer label
        answer = hdf5_file["answer_labels"][idx]
        answer = torch.from_numpy(np.asarray([answer])).long()
        hdf5_file.close()

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
            # obtain assignment label
            base_logits = io_utils.load_hdf5(self.base_logits_path, verbose=False)
            base_logits = torch.from_numpy(base_logits["base_logits"][idx])
            out.append(base_logits)
        out.append(answer)
        out.append(qst_id)

        return out


    def __len__(self):
        return len(self.json_file["image_filenames"])

    def get_samples(self, num_samples):
        """ Retrun sample data
        Returns: list of below items
            img (or feat): image (or feature)
            qst_label: question label
            qst_length: question length
            (assignments: pre-computed assignments)
            answer: answer label for questions
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

            random_prob = np.random.rand(1)[0]
            if random_prob < 0.001:
                sample_answers.append(cur_answer_label)
                samples.append([*sample[:-1], img_filename])
                cur_num_samples += 1
            else:
                # we get samples with different answers
                cur_answer_label = sample[-2][0]
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
        batch: list of two components [network inputs, image info]
            - network intpus: list of inputs [imgs, qst_labels, qst_lengths,
            (assignments), answers]
            - img_info: question ids or image filenames
    Returns: Sorted items in decreasing order of question length
        imgs: torch FloatTensor [batch_size, feat_dim, kernel_size, kernel_size].
        qst_labels: torch LongTensor [batch_size, padded_length].
        qst_lengths: list whose length is batch_size; valid length for each padded caption.
        (assignments): pre-computed assignments
        answers: torch LongTensor [batch_size]
    """
    # sort a batch by question length (descending order)
    batch.sort(key=lambda x: x[2], reverse=True)
    batch = [val for val in zip(*batch)]

    items = batch[:-1]
    img_info = batch[-1] # qst_id or img_filename
    return_vals = [items, img_info]

    # stacking tensors
    num_items = len(items)
    items[0] = torch.stack(items[0], 0) # images
    items[1] = torch.stack(items[1], 0).long() # question labels
    items[-1] = torch.stack(items[-1], 0).squeeze() # answers
    if num_items == 5:
        items[-2] = torch.stack(items[-2], 0) # for pre-computed logits of base models
    return return_vals

