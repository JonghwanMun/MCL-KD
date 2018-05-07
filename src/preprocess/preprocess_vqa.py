import os
import h5py
import json
import argparse
import numpy as np
from tqdm import tqdm

from src.dataset import vqa_dataset

def _GetArgumentParams():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target_splits", nargs='+', default=['train', 'val'],
                        help="List of target splits. Subset of 'train', 'val', 'test' can be used.")
    parser.add_argument("--vocab_path",
                        default="data/VQA_v2.0/preprocess/vocabulary/vqa_2000_vocab_train-val_raw.json",
                        help="Target vocabulary file used for preprocessing")
    parser.add_argument("--save_encoded_qa_dir",
                        default="data/VQA_v2.0/preprocess/encoded_qa",
                        help="Directory for saving encoded question answer data")
    parser.add_argument("--question_filter_option", default='only_questions',
                        help="Which questions will be encoded:"
                             "[only_questions|all_questions_with_answer_vocab|"
                             "questions_with_length|questions_shorter_or_equal_to]")
    parser.add_argument("--question_length", default=5, type=int,
                        help="Length of target questions. This option is enabled when "
                             "'question_filter_option' is equal to 'questions_with_length' or "
                             "'questions_shorter_or_equal_to'")
    parser.add_argument("--max_question_length", type=int, default=-1,
                        help="Max question length, -1 means obtaining from dataset")
    parser.add_argument("--use_right_align", action="store_true", default=False,
                        help="Left align processed question tokens. Default is right alignment")
    parser.add_argument("--use_zero_token", action="store_false", default=True,
                        help="Use zero token. Word tokens (including EMPTY) start from 0")
    params = parser.parse_args()
    print(json.dumps(vars(params), indent=4))
    return params

""" Encode question answer

Encode VQA dataset questions and answers for using with torch training code.
Args:
	vqa_data: Instance of VQADataset class. This instance is used to get
		tokenized questions
	wtoi: Word to integer label map. The integer labels start with 1 (not 0), to
		be used with torch.

Returns:
    encoded_qa: Dictionary containing following results:
    question_labels: Numpy array containing integer labels of question sentenes.
        The label index start from 1.
    question_length: Numpy array of question lengths.
    most_frequent_answer_labels: Numpy array containing integer labels of
        most frequent answer words. The label index start from 1.
    second_frequent_answer_labels: Numpy array containing integer labels of
        second most frequent answer words. The label index start from 1.
    all_answer_labels: Numpy array containing integer labels of all answer words
        which is collected from 10 annotators. The label index start from 1.
    question_ids: Question id.
    image_filenames: The image names for each question.
"""
def EncodeQuestionAnswer(questions, annotations, max_question_length, wtoi, \
                         atoi, use_right_align=False, use_zero_token=False, \
                         only_qst=False):
    assert (annotations == None or len(questions) == len(annotations)), \
        'The number of question and annotation should be same'

    num_examples = len(questions)
    if use_zero_token:
        question_labels = np.zeros((num_examples, max_question_length), dtype='int32')
    else:
        question_labels = np.ones((num_examples, max_question_length), dtype='int32')
    question_length = np.zeros(num_examples, dtype='int32')
    if not only_qst:
        most_frequent_answer_labels = np.zeros(num_examples, dtype='int32')
        all_answer_labels = np.zeros((num_examples, 10), dtype='int32')
        all_answer_labels_mask = np.zeros((num_examples, 10, 1), dtype='int32')
        second_frequent_answer_labels = []
    question_ids = []
    image_filenames = []
    num_only_mc_answer = 0
    num_second_answer = 0
    qst_word_set = wtoi.keys()
    ans_word_set = atoi.keys()
    itoa = {i:a for  a,i in atoi.items()}
    null_answer = itoa[len(wtoi)-1]
    for n, (q, a) in tqdm(enumerate(zip(questions, annotations))):
        question_length[n] = min(max_question_length, q['question_length'])
        if use_right_align:
            first_word_index = max_question_length - question_length[n]
        else:
            first_word_index = 0

        # Encode question labels
        for t, w in enumerate(q['tokenized_question']):
            if t == max_question_length: break
            if not w in qst_word_set:
                question_labels[n, first_word_index + t] = 1  # UNKNOWN word
            else:
                question_labels[n, first_word_index + t] = wtoi[w]
        question_ids.append(q['question_id'])

        image_filenames.append(q['image_filename'])

        if not only_qst:
            # Extract most frequent answer
            nth_most_ans = a['multiple_choice_answer']
            if not nth_most_ans in ans_word_set:
                # this is entered only when question_filter_option is only_questions
                most_frequent_answer_labels[n] = len(atoi)-1
                nth_most_ans = null_answer
            else:
                most_frequent_answer_labels[n] = atoi[nth_most_ans]

            answer_count = {}
            for ia, ith_answer_info in enumerate(a['answers']):
                ith_answer = ith_answer_info['answer']
                answer_count[ith_answer] = answer_count.get(ith_answer, 0) + 1
                # construct all answer labels and their mask
                if ith_answer in ans_word_set:
                    all_answer_labels[n, ia] = atoi[ith_answer]
                    all_answer_labels_mask[n, ia, 0] = 1
                else:
                    # TODO(jonghwan): think more nice way to deal with answers
                    # that are not in answer vocabulary
                    all_answer_labels[n, ia] = atoi[nth_most_ans]
            # Extract second frequent answer
            num_candidates = 0
            num_correct_answers = 0
            tmp_second = {'answer': 'NONE', 'p': 0.0}
            for k in answer_count.keys():
                if answer_count[k] > 2:  # think as correct answer when count > 2
                    num_correct_answers += answer_count[k]
            for k in answer_count.keys():
                if answer_count[k] > 2:  # think as correct answer when count > 2
                    num_candidates += 1
                    if (k != nth_most_ans) and (k in ans_word_set):
                        cur_p = answer_count[k] / num_correct_answers
                        if tmp_second['p'] < cur_p:
                            tmp_second = {'answer': atoi[k], 'p': cur_p}
            second_frequent_answer_labels.append(tmp_second)
            if num_candidates == 1:
                num_only_mc_answer += 1
            else:
                num_second_answer += 1

    if not only_qst:
        print(('Among {} answers: {}/{:.3f}(only most frequent answer) |' + \
               '{}/{:.3f}(second frequent answer)').format( \
                   len(annotations), num_only_mc_answer,
                   num_only_mc_answer * 100.0 / len(annotations), \
                   num_second_answer, num_second_answer * 100.0 / len(annotations)))

    # save the encoded questions and answers
    encoded_qa = {
        'question_labels': question_labels, \
        'question_length': question_length, \
        'question_ids': question_ids, \
        'image_filenames': image_filenames \
    }
    if not only_qst:
        encoded_qa['most_frequent_answer_labels'] = most_frequent_answer_labels
        encoded_qa['second_frequent_answer_labels'] = second_frequent_answer_labels
        encoded_qa['all_answer_labels'] = all_answer_labels
        encoded_qa['all_answer_labels_mask'] = all_answer_labels_mask

    return encoded_qa


""" Load vocabulary

Load vocabulary file containing vocabulary dict
Args:
	vocab_path: Path to vocabulary json file.
"""
def LoadVocabulary(vocab_path):
    print('Loading vocabulary from {}'.format(vocab_path))
    vocab_dict = json.load(open(vocab_path, 'r'))
    return vocab_dict

def main(params):

    # Check arguments
    if params.question_filter_option not in ['only_questions', 'all_questions_with_answer_vocab',
                                           'questions_with_length', 'questions_shorter_or_equal_to']:
        raise ValueError("Incorrect question_filter_option. Possible options are " +
                         "[only_questions|questions_with_length|questions_shorter_or_equal_to")
        if params.question_length < 0:
            raise ValueError("Incorrect question_length option. " +
                             "This option should be positive")

    vocab_dict = LoadVocabulary(params.vocab_path)

    print("Start preprocessing vqa dataset")
    dts = vqa_dataset.GetVQADataset(params.target_splits,
        skip_annotation=(params.question_filter_option == 'only_questions'))
    # Perform tokenization with vocabulary option.
    dts.LoadVocabulary(vocab_dict)

    vocabulary = vocab_dict['vocabulary']
    word_index = vocab_dict['word_index']

    wtoi = word_index['question']
    itow = {i:w for  w,i in wtoi.items()}
    atoi = word_index['answer']
    itoa = {i:a for  a,i in atoi.items()}

    print("Start encoding vqa dataset")
    # Sample questions where the corresponding answers are in answer vocabulary
    if params.question_filter_option == 'only_questions':
        only_qst = True
    elif params.question_filter_option == 'all_questions':
        # TODO:
        only_qst = False
    else:
        dts.SamplingQuestionsAnnotationsWithFrequentAnswers()
        only_qst = False

    # Encode question and answer.
    if params.question_filter_option == 'only_questions':
        question_subset_name = params.question_filter_option
        questions, annotations = dts.GetQuestionsAndAnnotations()
    elif params.question_filter_option == 'all_questions':
        question_subset_name = params.question_filter_option
        questions, annotations = dts.GetQuestionsAndAnnotations()
    elif params.question_filter_option == 'all_questions_with_answer_vocab':
        question_subset_name = params.question_filter_option
        questions, annotations = dts.GetQuestionsAndAnnotations()
    elif params.question_filter_option == 'questions_with_length':
        question_subset_name = "{}_{}".format(params.question_filter_option,
                                              params.question_length)
        questions, annotations = dts.GetQuestionsWithLength(params.question_length)
    elif params.question_filter_option == 'questions_shorter_or_equal_to':
        question_subset_name = "{}_{}".format(params.question_filter_option,
                                              params.question_length)
        questions, annotations = dts.GetShortQuestions(params.question_length)
    else:
        raise ValueError("Incorrect question_filter_option")

    # Encode QA
    if params.max_question_length == -1:
        max_question_length = max([q['question_length'] for q in questions])
    else:
        max_question_length = params.max_question_length
    encoded_qa = EncodeQuestionAnswer(questions, annotations, \
                                      max_question_length, \
                                      wtoi, atoi, \
                                      use_right_align=params.use_right_align, \
                                      use_zero_token=params.use_zero_token, \
                                      only_qst=only_qst)

    # Create target directory for saving encoded qa.
    save_encoded_qa_dir = os.path.join(params.save_encoded_qa_dir,
        params.vocab_path.split('/')[-1].replace('.json', ''), question_subset_name)
    if params.use_right_align:
        save_encoded_qa_dir += "_right_aligned"
    if params.use_zero_token:
        save_encoded_qa_dir += "_use_zero_token"
    save_encoded_qa_dir += "_max_qst_len_{}".format(max_question_length)
    if not os.path.isdir(save_encoded_qa_dir):
        print("Directory doesn't exist, create one: {}".format(save_encoded_qa_dir))
        os.makedirs(save_encoded_qa_dir)

    # Save file name
    save_filename = "qa_{}".format('-'.join(sorted(params.target_splits)))

    # Save processed numeric data with hdf5.
    print("Start saving question & answer labels")
    save_hdf5_path = os.path.join(save_encoded_qa_dir, "{}.h5".format(save_filename))
    f = h5py.File(save_hdf5_path, "w")
    f.create_dataset("question_labels", dtype='int32', data=encoded_qa['question_labels'])
    f.create_dataset("question_length", dtype='int32', data=encoded_qa['question_length'])
    if not only_qst:
        f.create_dataset("most_frequent_answer_labels", dtype='int32',
                         data=encoded_qa['most_frequent_answer_labels'])
        f.create_dataset("all_answer_labels", dtype='int32', \
                         data=encoded_qa['all_answer_labels'])
        f.create_dataset("all_answer_labels_mask", dtype='int32', \
                         data=encoded_qa['all_answer_labels_mask'])
    f.close()
    print("Saving is done: {}".format(save_hdf5_path))

    # Save processes list data with json.
    print("Start saving vocabulary and other information")
    save_json_path = os.path.join(save_encoded_qa_dir, "{}.json".format(save_filename))
    out = {}
    out['vocab_info'] = vocab_dict['info']
    out['splits'] = params.target_splits
    out['wtoi'] = wtoi
    out['itow'] = itow
    out['atoi'] = atoi
    out['itoa'] = itoa
    out['question_ids'] = encoded_qa['question_ids']
    out['image_filenames'] = encoded_qa['image_filenames']
    if not only_qst:
        out['second_frequent_answer_labels'] = encoded_qa['second_frequent_answer_labels']
    json.dump(out, open(save_json_path, 'w'))
    print("Saving is done: {}".format(save_json_path))


if __name__ == "__main__":
    params = _GetArgumentParams()
    main(params)
