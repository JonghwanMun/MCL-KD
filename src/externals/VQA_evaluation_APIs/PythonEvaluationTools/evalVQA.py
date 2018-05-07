# coding: utf-8

from __future__ import print_function, division
import sys
dataDir = 'src/externals/VQA_evaluation_APIs'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
sys.path.append('%s/PythonEvaluationTools' %(dataDir))
import os
import json
import random
import argparse
import skimage.io as io
import matplotlib.pyplot as plt

from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prediction_json_path", required=True,
        help="Path to prediction result file")
    parser.add_argument("--small_set", action='store_true', default=False,
        help="options to evaluate subset")
    parser.add_argument("--verbose", action='store_true', default=False,
        help="verbose")
    args = parser.parse_args()
    print(args)

    # set up file names and paths
    annFile  = 'data/VQA_v2.0/annotations/v2_mscoco_val2014_annotations.json'
    quesFile = 'data/VQA_v2.0/annotations/v2_OpenEnded_mscoco_val2014_questions.json'

    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    num_all_examples = len(vqa.qa)
    vqaRes = vqa.loadRes(args.prediction_json_path, quesFile, args.small_set)
    num_eval_examples = len(vqa.qa)

    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval = VQAEval(vqa, vqaRes, n=2)

    # evaluate results
    if args.small_set:
        # If you have a list of question ids on which you would like to evaluate
        # your results, pass it as a list to below function. By default it uses
        # all the question ids in annotation file.
        resAnns = json.load(open(args.prediction_json_path))
        qstIds = [int(qst['question_id']) for qst in resAnns]
        acc_per_qstid = vqaEval.evaluate(qstIds)
        print("Accuracy on subset is: %.02f" % (vqaEval.accuracy['overall']))
        print("Accuracy on all examples is: %.02f\n" %
              (vqaEval.accuracy['overall']*num_eval_examples/num_all_examples))
    else:
        acc_per_qstid = vqaEval.evaluate()
        print("Accuracy is: %.02f" % (vqaEval.accuracy['overall']))


    # print accuracies
    if args.verbose:
        print("Per Question Type Accuracy is the following:")
        for quesType in vqaEval.accuracy['perQuestionType']:
            print( "%s : %.02f" %(quesType,
                               vqaEval.accuracy['perQuestionType'][quesType]))
        print("\n")
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")

if __name__ == "__main__":
    main()
