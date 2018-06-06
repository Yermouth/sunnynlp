# SUNNYNLP

This repository contains the code for our paper:
[SUNNYNLP at SemEval-2018 Task 10: A Support-Vector-Machine-Based Method for Detecting Semantic Difference using Taxonomy and Word Embedding Features](http://aclweb.org/anthology/S18-1118)

Task description: [SemEval 2018 Task 10 -- Capturing Discriminative Attributes](https://competitions.codalab.org/competitions/17326)

Our Support-Vector-Machine(SVM)-based system combines features extracted from pre-trained embeddings and statistical information from Probase to detect semantic difference of concepts pairs.

## Requirements
### Python and packages
We recommend using a separate Python 3.6 environment to install packages. All packages required are listed in `requirements.txt`. You can install them using pip:
```
pip install -r requirements.txt
```
As our system is using the English model in spaCy, run
```
python -m spacy download en
```
to install the language model required

### Data
- Pre-trained word vectors:
  - [Word2Vec](https://code.google.com/archive/p/word2vec/)
  - [GloVe](https://nlp.stanford.edu/projects/glove/)
  - [FastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)

- [Our spaCy parsed version of Probase](https://github.com/Yermouth/spacy-probase) or the original [Probase](https://www.microsoft.com/en-us/research/project/probase/)


## Basic usage
- Edit the path for Probase, pre-trained vectors, output path, etc. according to the instructions in [`./config/configuration.yml`](config/configuration.yml).

- Run the main program in root directory to generate predictions based on your configuration:
```
$ python src/main.py config/configuration-1.yml
```

- Run the official script to evaluate the predictions in the directory you have specified and save scores in `./score/`
```
$ ./official-evaluation.sh ./prediction/configuration-1
./prediction/prediction-final/dev-5folds-FastText-dtc.txt
3 ./score/all-score.txt
./prediction/prediction-final/dev-5folds-FastText-LinearSVC.txt
5 ./score/all-score.txt
```

## References
If you find our work useful, please cite our work.

```
@inproceedings{lai2018sunnynlp,
  title={SUNNYNLP at SemEval-2018 Task 10: A Support-Vector-Machine-Based Method for Detecting Semantic Difference using Taxonomy and Word Embedding Features},
  author={Lai, Sunny and Leung, Kwong Sak and Leung, Yee},
  booktitle={Proceedings of The 12th International Workshop on Semantic Evaluation},
  pages={741--746},
  year={2018}
}
```

If you use the code, please cite according to the [hyperwords repository](https://bitbucket.org/omerlevy/hyperwords)

If you have used our spaCy parsed version of Probase, please cite according to the [Probase official website](https://www.microsoft.com/en-us/research/project/probase/)

