import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


class Evaluation(object):
    def __init__(self, config, dataset):
        self.random_seed = config.random_seed
        self.split_configs = config.split_configs
        self.cross_validation_configs = config.cross_validation_configs
        self.feature_configs = config.feature_configs
        self.classifier_configs = config.classifier_configs
        self.output_path = config.output_path
        self.models = {}
        self.dataset = dataset
        self.init_model()

    def init_model(self):
        self.models = {
            'logistic': LogisticRegression(random_state=self.random_seed),
            'dtc': DecisionTreeClassifier(random_state=self.random_seed),
            'svm-rbf': SVC(kernel='rbf', random_state=self.random_seed),
            'rf': RandomForestClassifier(random_state=self.random_seed),
            'LinearSVC': LinearSVC(random_state=self.random_seed),
        }

    def full_evaluation(self, X, y):
        # Cross Validation
        os.makedirs(self.output_path, exist_ok=True)
        for s in self.cross_validation_configs:
            folds, uses = s['folds'], s['uses']
            for f in self.feature_configs:
                folds = s['folds']
                config_description = '||\t||'.join([
                    '||Training: ' + ' + '.join(uses),
                    'folds: ' + str(folds),
                    'Feature: ' + ' + '.join(f) + '||'
                ])
                config_name = '-'.join(['-'.join(uses), str(folds)+'folds', '_'.join(f)])
                self.cross_validate((
                    config_description,
                    config_name,
                    self._combine_X(X, uses, f),
                    self._combine_y(y, uses),
                    folds,
                ), self.random_seed)

        # Train Test split
        for sc in self.split_configs:
            for fc in self.feature_configs:
                config_description = '||\t||'.join([
                    '||Training: ' + ' + '.join(sc['train']),
                    'Testing: ' + ' + '.join(sc['test']),
                    'Feature: ' + ' + '.join(fc) + '||'
                ])
                config_name = '-'.join(
                    ['-'.join(sc['train'])] +
                    ['-'.join(sc['test'])] +
                    ['_'.join(fc)]
                )
                self.evaluate((
                    config_description,
                    config_name,
                    self._combine_X(X, sc['train'], fc),
                    self._combine_y(y, sc['train']),
                    self._combine_X(X, sc['test'], fc),
                    self._combine_y(y, sc['test']),
                ), self.random_seed)

    def cross_validate(self, config, rs, init=True):
        config_descripiton, config_name, X, y, folds = config
        if init:
            self.init_model()
        for clsname, model in self.models.items():
            if clsname in self.classifier_configs:
                print(config_descripiton)
                print('File name: ', config_name + '-' + clsname + '.txt')
                print('Classifier: ', clsname)
                prediction = cross_val_predict(model, X, y, cv=folds)
                print('accuracy: ', accuracy_score(y, prediction))
                print('f1 score: ', f1_score(y, prediction))
                print('cofusion matrix:')
                print(confusion_matrix(y, prediction), '\n')
                with open(os.path.join(self.output_path, config_name + '-' + clsname + '.txt'), 'w') as wf:
                    reference = self.dataset.test
                    if config_name.startswith("dev") or config_name.startswith("valid"):
                        reference = self.dataset.dev
                    elif config_name.startswith("train"):
                        reference = self.dataset.train

                    for x, p in zip(reference, prediction):
                        wf.write(','.join(x[:3]+[str(p)])+'\n')

    def evaluate(self, config, rs, init=True):
        config_descripiton, config_name, X1, y1, X2, y2 = config
        if init:
            self.init_model()
        for clsname, model in self.models.items():
            if clsname in self.classifier_configs:
                print(config_descripiton)
                print('File name: ', config_name + '-' + clsname + '.txt')
                print('Classifier: ', clsname)
                model.fit(X1, y1)
                prediction = model.predict(X2)
                print('accuracy: ', accuracy_score(y2, prediction))
                print('f1 score: ', f1_score(y2, prediction))
                print('cofusion matrix:')
                print(confusion_matrix(y2, prediction), '\n')
                prediction_output_path = os.path.join(self.output_path, config_name + '-' + clsname + '.txt')
                with open(prediction_output_path, 'w') as wf:
                    for x, p in zip(self.dataset.test, prediction):
                        wf.write(','.join(x[:3]+[str(p)])+'\n')

    def _concat_data(self, X1, X2):
        return [np.concatenate([x1, x2]) for x1, x2 in zip(X1, X2)]

    def _combine_X(self, data, split, feature):
        combined = []
        all_combined = []
        if len(split) == 1 and len(feature) == 1:
            return data[split[0]][feature[0]]

        elif len(split) >= 2 and len(feature) == 1:
            for s in split:
                combined += data[s][feature[0]]
            return combined

        elif len(split) == 1 and len(feature) >= 2:
            combined = data[split[0]][feature[0]]
            for f in feature[1:]:
                combined = self._concat_data(combined, data[split[0]][f])
            return combined

        elif len(split) >= 2 and len(feature) >= 2:
            for s in split:
                all_combined += data[s][feature[0]]

            for f in feature[1:]:
                combined = []
                for s in split:
                    combined += data[s][f]
                all_combined = self._concat_data(all_combined, combined)
            return all_combined

    def _combine_y(self, data, split):
        if len(split) == 1:
            return data[split[0]]

        elif len(split) >= 2:
            combined = []
            for s in split:
                combined += data[s]
            return combined
        else:
            print('error combining y')
