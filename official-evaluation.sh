#!/bin/bash
mkdir -p ./score
echo "Official evaluation F1-scores" > ./score/all-score.txt;
for filename in $1/*.txt; do
    if [[ $filename = *"train-"*"fold"* ]]; then
        cp ./ref/train-truth.txt ./ref/truth.txt
    elif [[ $filename = *"dev-"*"fold"* ]]; then
        cp ./ref/validation-truth.txt ./ref/truth.txt
    else
        cp ./ref/test-truth.txt ./ref/truth.txt
    fi
    echo "$filename";
    cp "$filename" res/answer.txt;
    python ./official-evaluation.py ./ score/ > tmp.txt;
    echo $filename >> ./score/all-score.txt;
    cat ./score/scores.txt >> ./score/all-score.txt;
    wc -l ./score/all-score.txt;
done
mkdir -p $(dirname ./score/${1}.score)
cp ./score/all-score.txt ./score/${1}.score

rm tmp.txt;
