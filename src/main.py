from argparse import ArgumentParser

from configuration import Configuration
from data import Data
from evaluation import Evaluation
from feature import Feature
from nlp import NLP
from probase import Probase


def main(args):
    # Load configuration
    config = Configuration(args.yaml_path)

    print("Loading Probase...")
    probase = Probase(config)

    print("Loading dataset...")
    dataset = Data(config)

    print("Loading NLP utility...")
    nlp = NLP('en')

    print("Loading feature extractor...")
    features = Feature(config, probase, nlp=nlp)

    print("Extracting vector features")
    features.extract_vector_features(dataset)

    print("Extracting statistical vector features")
    features.extract_statistical_features(dataset)

    print("Evaluating clasifiers")
    ev = Evaluation(config, dataset)
    ev.full_evaluation(features.X, features.y)


if __name__ == "__main__":
    parser = ArgumentParser(description='')
    parser.add_argument('yaml_path')
    args = parser.parse_args()
    main(args)
