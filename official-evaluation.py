#!/usr/bin/env python
import csv
import os
import os.path
import sys


def f1_score(evaluation):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in evaluation.values():
            if i[0] == i[1] and i[1] == 1: 
                tp = tp+1
            if i[0] == i[1] and i[1] == 0: 
                tn = tn+1
            elif i[0] != i[1] and i[1] == 1: 
                fp = fp+1
            elif i[0] != i[1] and i[1] == 0: 
                fn = fn+1
    f1_positives = 0.0
    f1_negatives = 0.0
    if tp>0:
        precision=float(tp)/(tp+fp)
        recall=float(tp)/(tp+fn)
        f1_positives = 2*((precision*recall)/(precision+recall))
    if tn>0:
        precision=float(tn)/(tn+fn)
        recall=float(tn)/(tn+fp)
        f1_negatives = 2*((precision*recall)/(precision+recall))
    if f1_positives and f1_negatives:
        f1_average = (f1_positives+f1_negatives)/2.0
        return f1_average
    else:
        return 0


# as per the metadata file, input and output directories are the arguments
[_, input_dir, output_dir] = sys.argv

# unzipped submission data is always in the 'res' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
submission_file_name = 'answer.txt'
submission_dir = os.path.join(input_dir, 'res')
submission_path = os.path.join(submission_dir, submission_file_name)
if not os.path.exists(submission_path):
    message = "Expected submission file '{0}', found files {1}"
    sys.exit(message.format(submission_file_name, os.listdir(submission_dir)))
evaluation = {}

# unzipped reference data is always in the 'ref' subdirectory
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
with open(os.path.join(input_dir, 'ref', 'truth.txt')) as truth_file:
    reader = csv.reader(truth_file)
    for row in reader:
        if len(row) != 4:
            message = "Truth file has row of length "+str(len(row))+", len 3 needed."
            sys.exit(message)
        if type(int(row[3])) != int:
            message = "Value is not of type int."
            sys.exit(message)
        k = ','.join(row[:3])
        print(k)
        v = [int(row[3])]
        evaluation[k] = v

with open(submission_path) as submission_file:
    reader = csv.reader(submission_file)
    for row in reader:
        if len(row) != 4:
            message = "Submission file has row of length "+str(len(row))+", len 3 needed."
            sys.exit(message)
        if type(int(row[3])) != int:
            message = "Value is not of type int."
            sys.exit(message)
        k = ','.join(row[:3])
        if k not in evaluation.keys():
            message = "Entry "+k+" does not exist in reference file."
            sys.exit(message)
        else:
            v = int(row[3])
            evaluation[k].append(v)

if not evaluation:
    message = "evaluation empty"
    sys.exit(message)

for k,v in evaluation.items():
    if len(v) != 2:
        if len(v)<2:
            message = "Entry "+k+" is missing in submission file."
        elif len(v)>2:
            message = "Entry "+k+" has duplicate entries in submission file."
        sys.exit(message)

# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions


with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
    score = f1_score(evaluation)
    output_file.write("correct:{0}\n".format(score))

with open(os.path.join(output_dir, 'stderr.txt'), 'w') as stderr_file:
    score = f1_score(evaluation)
    stderr_file.write('output from scoring function ='+str(score)+'\n')
    stderr_file.write("correct:{0}\n".format(score))
