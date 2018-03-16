#! /bin/bash

if [ $2 -eq 1 ]
then
	python -m externals.VQA_evaluation_APIs.PythonEvaluationTools.evalVQA \
		--prediction_json_path $1 \
		--small_set
else
	python -m externals.VQA_evaluation_APIs.PythonEvaluationTools.evalVQA \
		--prediction_json_path $1
fi
