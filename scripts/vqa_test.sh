#! /bin/bash

if [ $2 -eq 1 ]
then
	python -m src.externals.VQA_evaluation_APIs.PythonEvaluationTools.evalVQA \
		--prediction_json_path $1 \
		--small_set
else
	python -m src.externals.VQA_evaluation_APIs.PythonEvaluationTools.evalVQA \
		--prediction_json_path $1
fi
