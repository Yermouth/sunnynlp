import math
from itertools import combinations
from math import log

import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cosine

from gensim.models import KeyedVectors
from tqdm import tqdm


class Feature(object):
    def __init__(self, config, probase, nlp=None):
        self.pretrain_embeddings = config.pretrain_embeddings
        self.data_type = config.data_type
        self.X = {d:{} for d in self.data_type}
        self.y = {}
        self.vectors = []
        self.load_vectors_from_path(self.pretrain_embeddings, clear=True)
        self.nlp = nlp
        self.pb = probase

    def extract_vector_features(self, dataset):
        for embed, vector in tqdm(zip(self.pretrain_embeddings, self.vectors)):
            for dt in self.data_type:
                _X, _y = self.vector_features(getattr(dataset, dt), vector)
                self.X[dt][embed['name']], self.y[dt] = _X, _y

    def extract_statistical_features(self, dataset):
        for dt in tqdm(self.data_type):
            _X, _y = self.statistical_features(getattr(dataset, dt))
            self.X[dt]['Probase'], self.y[dt] = _X, _y

    def load_vectors_from_path(self, pretrain_embeddings, clear=True):
        if clear:
            self.vectors = []
        for pe in pretrain_embeddings:
            embed_name, embed_path, is_binary = pe['name'], pe['path'], pe['is_bin']
            print('Loading pretrain embeddings:', embed_name)
            if is_binary:
                self.vectors.append(KeyedVectors.load_word2vec_format(embed_path, binary=True))
            else:
                self.vectors.append(KeyedVectors.load_word2vec_format(embed_path))

    def wr(self, vector, w):
        """Word (Vector) Representation"""
        if w in vector.wv:
            return vector.wv[w]
        elif self.nlp and self.nlp.lemma(w) in vector.wv:
            return vector.wv[self.nlp.lemma(w)]
        else:
            return np.ones(300)

    def vector_features(self, dataset, vector):
        X = []
        y = []
        for d in dataset:
            y.append(d[-1])

            ## Original vector
            original_vectors = np.concatenate([self.wr(vector, w) for w in d[:-1]])

            ## Pairwise L1 norm
            w1w2_norm = self._wrap_np([self.vec_diff_norm(d[0],d[1],vector)])
            w1attr_norm = self._wrap_np([self.vec_diff_norm(d[0],d[2],vector)])
            w2attr_norm = self._wrap_np([self.vec_diff_norm(d[1],d[2],vector)])

            ## Pairwise Cosine Distance
            w1w2_cos = self._wrap_np([self.vec_cosine(d[0],d[1],vector)])
            w1attr_cos = self._wrap_np([self.vec_cosine(d[0],d[2],vector)])
            w2attr_cos = self._wrap_np([self.vec_cosine(d[1],d[2],vector)])

            X.append(np.concatenate([
                original_vectors,
                w1w2_norm,
                w1attr_norm,
                w2attr_norm,
                w1w2_cos,
                w1attr_cos,
                w2attr_cos,
            ]))
        return X, y

    def _wrap_np(self, values):
        return np.array(values, dtype=np.float)

    def vec_cosine(self, w1, w2, vector):
        return cosine(self.wr(vector, w1), self.wr(vector, w2))

    def vec_diff_norm(self, w1, w2, vector, n=1):
        return LA.norm(self.wr(vector, w1) - self.wr(vector, w2), n)

    def statistical_features(self, dataset):
        def safe_log(x):
            return math.log(x) if x > 0 else 0

        X = []
        y = []
        for d in dataset:
            y.append(d[-1])
            w1attr_stat = [x for x in map(safe_log, self.pb.statistical_features(d[2], d[0]))]
            w2attr_stat = [x for x in map(safe_log, self.pb.statistical_features(d[2], d[1]))]
            X.append(self._wrap_np(w1attr_stat + w2attr_stat))
        return X, y

