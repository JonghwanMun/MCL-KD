import os
import json
import argparse

import h5py
import numpy as np

from src.dataset import clevr_dataset
from src.utils import io_utils

""" Encode question answer

Encode CLEVR dataset questions and answers for using with torch training code.
Args:
	clevr_data: Instance of ClevrDataset class. This instance is used to get
		tokenized questions
	wtoi: Word to integer label map. The integer labels start with 1 (not 0), to
		be used with torch.
	atoi: Answer to integer label map. Even though default answer label shares
		word index with question vocabulary, contiguous labels, which start from 1
		and label indexes are contiguous, uses atoi index map.
	use_right_align: Whether to align questions right. Default is left alignment.
	use_zero_token: Empty question tokens are 0, not 1. Default is true.

Returns:
	encoded_qa: Dictionary containing following results:
		question_labels: Numpy array containing integer labels of question
			sentenes. The label index start from 1.
		question_length: Numpy array of question lengths.
		answer_labels: Numpy array containing integer labels of answer words. The
			label index start from 1.
		question_family: Numpy array of question family indexes, which starts
			from 0.
		question_ids: Question id containing the split and question number.
		image_filenames: The image names for each question.
"""
def EncodeQuestionAnswer(questions, max_question_length, wtoi, atoi,
                         use_right_align=False, use_zero_token=True):

    num_examples = len(questions)
    if use_zero_token:
        question_labels = np.zeros((num_examples, max_question_length), dtype="int32")
    else:
        question_labels = np.ones((num_examples, max_question_length), dtype="int32")

    # initialize data
    question_ids = []
    image_filenames = []
    question_length = np.zeros(num_examples, dtype="int32")
    answer_labels = np.zeros(num_examples, dtype="int32")
    question_family = np.zeros(num_examples, dtype="int32")

    # convert questions & answers to label
    qst_word_list = wtoi.keys()
    for n, q in enumerate(questions):
        question_length[n] = min(max_question_length, len(q["tokenized_question"]))
        if use_right_align:
            first_word_index = max_question_length - question_length[n]
        else:
            first_word_index = 0

        for t, w in enumerate(q["tokenized_question"]):
            if t == max_question_length: break
            if not w in qst_word_list:
                question_labels[n, first_word_index + t] = 1  # UNKNOWN word
            question_labels[n, first_word_index + t] = wtoi[w]

        answer_labels[n] = atoi[q["answer"]]
        question_family[n] = q["question_family_index"]
        question_ids.append("{}_{}".format(q["split"], q["question_index"]))
        image_filenames.append(os.path.join(q["split"], q["image_filename"]))

    encoded_qa = {}
    encoded_qa["question_labels"] = question_labels
    encoded_qa["question_length"] = question_length
    encoded_qa["answer_labels"] = answer_labels
    encoded_qa["question_family"] = question_family
    encoded_qa["question_ids"] = question_ids
    encoded_qa["image_filenames"] = image_filenames
    return encoded_qa

def EncodeSceneLabels(scenes, wtoi, itow):
	num_examples = len(scenes)
	entity_existence = np.zeros((num_examples, max(itow.keys())), dtype="int32")
	for n, scene in enumerate(scenes):
		entities = list(set(scene["entity"]))
		for entity in entities:
			# Index for entity tokens starts from 1, so we have to subtract 1.
			entity_existence[n][wtoi[entity]-1] = 1
	encoded_scenes = {
		"entity_existence": entity_existence,
	}
	return encoded_scenes

def _GetArgumentParams():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--interactive" , action="store_true", default=False,
                        help="Run the script in an interactive mode")
    parser.add_argument("--target_splits", nargs="+", default=["train"],
                        help="List of target splits. Subset of 'train', 'val', 'test' can be used.")
    parser.add_argument("--vocab_path",
                        default="data/CLEVR_v1.0/preprocess/vocabulary/clevr_vocab_train_raw.json",
                        help="Target vocabulary file used for preprocessing")
    parser.add_argument("--save_encoded_qa_dir",
                        default="data/CLEVR_v1.0/preprocess/encoded_qa",
                        help="Directory for saving encoded question answer data")
    parser.add_argument("--question_length_filter_option", default=None,
                        help="Option for selecting questions based on length [lt|eq|gt]")
    parser.add_argument("--question_length", default=None, type=int,
                        help="Length of question. This option is jointly used with quesiton"
                        + "length filter option.")
    parser.add_argument("--question_family_index_list", nargs="+", default=None,
                        type=int, help="List of target question family index. None not to specify")
    parser.add_argument("--max_question_length", type=int, default=-1,
                        help="Max question length, -1 means obtaining from dataset")
    parser.add_argument("--use_right_align", action="store_true", default=False,
                        help="Left align processed question tokens. Default is right alignment")
    parser.add_argument("--use_zero_token", action="store_true", default=False,
                        help="Do not use zero token. Word tokens (including EMPTY) start from 1")
    parser.add_argument("--contain_scene_labels", action="store_true",
                        default=False, help="Contain scene labels in the data")
    parser.add_argument("--dataset_config_path", default="./data/clevr_path.yml",
                        help="Path to dataset config file: only for CLEVR")
    params = vars(parser.parse_args())
    print (json.dumps(params, indent=4))
    return params

def main(params):

    # Validate parameters
    if params["question_length_filter_option"] not in ["lt", "eq", "gt", None]:
        raise ValueError("Incorrect question_length_filter_option. [lt|eq|gt]")
    if params["question_length"] != None and params["question_length"] < 0:
        raise ValueError("Incorrect question_length option. " + "This option should be positive")
    if params["question_length"] != None and \
            params["question_length_filter_option"] not in ["lt", "eq", "gt"]:
        raise ValueError("The question_length_filter option is not set" +\
                         " while question_length is set")
    if params["question_family_index_list"] != None and\
            type(params["question_family_index_list"]) != list:
        raise ValueError("The type of question_family_index_list should be None or list")

    print ("Start preprocessing clevr dataset")
    clv = clevr_dataset.GetClevrDataset(splits=params["target_splits"],
                                        clevr_config=params["dataset_config_path"])

    vocab_dict = io_utils.load_json(params["vocab_path"])

    # Perform tokenization with vocabulary option.
    clv.ConstructVocabulary(vocab_option=vocab_dict["info"]["vocab_option"])

    vocabulary = vocab_dict["vocabulary"]
    word_index = vocab_dict["word_index"]

    wtoi = word_index["question"]
    itow = {i:w for  w,i in wtoi.items()}
    atoi = word_index["answer"]
    itoa = {i:a for  a,i in atoi.items()}

    # Encode question and answer.
    if params["question_family_index_list"] == None:
        if params["question_length"] == None:
            question_subset_name = "all_questions"
            questions = clv.questions
        else:
            if params["question_length_filter_option"] == "lt":
                question_subset_name = "lt_{}".format(params["question_length"])
                questions = clv.GetQuestionsWithFilter(
                    question_length_filter = lambda x: x < params["question_length"])
            elif params["question_length_filter_option"] == "eq":
                question_subset_name = "eq_{}".format(params["question_length"])
                questions = clv.GetQuestionsWithFilter(
                    question_length_filter = lambda x: x == params["question_length"])
            elif params["question_length_filter_option"] == "gt":
                question_subset_name = "gt_{}".format(params["question_length"])
                questions = clv.GetQuestionsWithFilter(
                    question_length_filter = lambda x: x > params["question_length"])

    elif type(params["question_family_index_list"]) == list:
        if params["question_length"] == None:
            question_subset_name = "family_{}".format(
                "_".join([str(idx)
                          for idx in sorted(params["question_family_index_list"])]))
            questions = clv.GetQuestionsWithFilter(
                question_family_index_filter =
                lambda x: x in params["question_family_index_list"])
        else:
            if params["question_length_filter_option"] == "lt":
                question_subset_name = "family_{}_lt_{}".format(
                    "_".join([str(idx)
                              for idx in sorted(params["question_family_index_list"])]),
                    params["question_length"])
                questions = clv.GetQuestionsWithFilter(
                    question_length_filter = lambda x: x < params["question_length"],
                    question_family_index_filter =
                    lambda x: x in params["question_family_index_list"])
            elif params["question_length_filter_option"] == "eq":
                question_subset_name = "family_{}_eq_{}".format(
                    "_".join([str(idx)
                              for idx in sorted(params["question_family_index_list"])]),
                    params["question_length"])
                questions = clv.GetQuestionsWithFilter(
                    question_length_filter = lambda x: x == params["question_length"],
                    question_family_index_filter =
                    lambda x: x in params["question_family_index_list"])
            elif params["question_length_filter_option"] == "gt":
                question_subset_name = "family_{}_gt_{}".format(
                    "_".join([str(idx)
                              for idx in sorted(params["question_family_index_list"])]),
                    params["question_length"])
                questions = clv.GetQuestionsWithFilter(
                    question_length_filter = lambda x: x > params["question_length"],
                    question_family_index_filter =
                    lambda x: x in params["question_family_index_list"])

    # Encode QA
    if params["max_question_length"] == -1:
        max_question_length = clv.GetMaxQuestionLength()
    else:
        max_question_length = params["max_question_length"]

    encoded_qa = EncodeQuestionAnswer(questions, max_question_length,
                                      wtoi, atoi, use_right_align=params["use_right_align"],
                                      use_zero_token=params["use_zero_token"])

    if params["contain_scene_labels"]:
        # Encode scene entity classification
        scenes = clv.GetScenesByQuestions(questions, use_vocabulary=True, use_entity=True)
        encoded_scenes = EncodeSceneLabels(scenes, wtoi, itow)

    # Create target directory for saving encoded qa.
    save_encoded_qa_dir = os.path.join(params["save_encoded_qa_dir"],
            "vocab_{}_{}".format("-".join(sorted(vocab_dict["info"]["splits"])),
            vocab_dict["info"]["vocab_option"]), question_subset_name)
    if params["use_right_align"]:
        save_encoded_qa_dir += "_right_aligned"
    if params["use_zero_token"]:
        save_encoded_qa_dir += "_use_zero_token"
    if params["contain_scene_labels"]:
        save_encoded_qa_dir += "_with_scene_labels"
    save_encoded_qa_dir += "_max_qst_len_{}".format(params["max_question_length"])
    if not os.path.isdir(save_encoded_qa_dir):
        print ("Directory doesn't exist, create one: {}".format( save_encoded_qa_dir))
        os.makedirs(save_encoded_qa_dir)

    # Save file name
    save_filename = "qa_{}".format("-".join(sorted(params["target_splits"])))

    # Save processed numeric data with hdf5.
    # TODO: save as npz
    save_hdf5_path = os.path.join(save_encoded_qa_dir,
                                  "{}.h5".format(save_filename))
    f = h5py.File(save_hdf5_path, "w")
    f.create_dataset("question_labels", dtype="int32", data=encoded_qa["question_labels"])
    f.create_dataset("question_length", dtype="int32", data=encoded_qa["question_length"])
    f.create_dataset("answer_labels", dtype="int32", data=encoded_qa["answer_labels"])
    f.create_dataset("question_family", dtype="int32", \
                         data=encoded_qa["question_family"])
    if params["contain_scene_labels"]:
        f.create_dataset("entity_existence", dtype="int32", data=encoded_scenes["entity_existence"])
    f.close()
    print ("Saving is done: {}".format(save_hdf5_path))

    # Save processes list data with json.
    save_json_path = os.path.join(save_encoded_qa_dir, "{}.json".format(save_filename))
    out = {}
    out["vocab_info"] = vocab_dict["info"]
    out["splits"] = params["target_splits"]
    out["wtoi"] = wtoi
    out["itow"] = itow
    out["atoi"] = atoi
    out["itoa"] = itoa
    out["question_ids"] = encoded_qa["question_ids"] # "split_qid"
    out["image_filenames"] = encoded_qa["image_filenames"]
    json.dump(out, open(save_json_path, "w"))
    print ("Saving is done: {}".format(save_json_path))


if __name__ == "__main__":
	params = _GetArgumentParams()
	if not params["interactive"]:
		main(params)
