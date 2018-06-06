import os
import pickle
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix

from tqdm import tqdm


class Probase(object):
    def __init__(self, config):
        self.config = config
        if not os.path.exists(config.model_dir):
            print('Previous probase model not found, creating new model')
            self.lemma_dir = config.lemma_dir
            self.init_attributes()
            print('Loading lemma from directory')
            self.load_lemma_from_directory(self.lemma_dir)
            print('Creating index dict')
            self.index_dict()
            print('Counting cooccur')
            self.count_occurrences()
            print('Saving model')
            self.save_model(config.model_dir)
            print('Model saved')
        else:
            print('Loading model from ' + config.model_dir)
            self.load_model(config.model_dir)

    def init_attributes(self):
        self.lemma = []
        self.count = []
        self.head_counter = Counter()
        self.tail_counter = Counter()
        self.ignore_attributes = [
            'lemma', 'head_counter', 'tail_counter', 'partitions'
        ]
        self.partition_types = ['1to1', 'Nto1', '1toN', 'NtoN']
        self.ind_freq_types = ['head', 'tail']
        self.sym_co_freq_types = ['head_head', 'tail_tail']
        self.asym_co_freq_types = ['head_tail', 'tail_head']
        self.co_freq_types = self.sym_co_freq_types + self.asym_co_freq_types
        self.freq_types = self.ind_freq_types + self.co_freq_types
        self.partitions = {p: {'ix': []} for p in self.partition_types}
        self.num_of_data = 0

    def load_lemma_from_directory(self, d):
        files = sorted(os.listdir(d))
        for file in tqdm(files):
            self.load_lemma_with_count(d + file)

    def load_lemma_with_count(self, file):
        def add_lemma_index_to_partition(head_words, tail_words):
            # 1 to 1
            if len(head_words) == 1 and len(tail_words) == 1:
                self.partitions['1to1']['ix'].append(self.num_of_data)
            # N to 1
            elif len(head_words) > 1 and len(tail_words) == 1:
                self.partitions['Nto1']['ix'].append(self.num_of_data)
            # 1 to N
            elif len(head_words) == 1 and len(tail_words) > 1:
                self.partitions['1toN']['ix'].append(self.num_of_data)
            # N to N
            elif len(head_words) > 1 and len(tail_words) > 1:
                self.partitions['NtoN']['ix'].append(self.num_of_data)
            self.num_of_data += 1

        with open(file, 'r') as rf:
            for line in rf:
                tail, head, count = line.strip().split('\t')
                count = int(count)
                self.count.append(count)
                head_words = head.split()
                for word in head_words:
                    self.head_counter[word] += count
                tail_words = tail.split()
                for word in tail_words:
                    self.tail_counter[word] += count
                self.lemma.append([head_words, tail_words])
                add_lemma_index_to_partition(head_words, tail_words)

    def index_dict(self):
        vocab_counters = [self.head_counter, self.tail_counter]
        vocabs = ['<@!>']
        for vc in vocab_counters:
            vocabs += list(vc.keys())
        vocabs = sorted(list(set(vocabs)))
        self.wi = {w: i for i, w in enumerate(vocabs)}
        self.iw = {i: w for i, w in enumerate(vocabs)}
        self.vocab_size = len(self.wi)

    def order(self, x, y):
        return (x, y) if x < y else (y, x)

    def count_freq(self, lemma, count):
        def asymmetric_count(s1, s2, cnt, counter):
            for h in s1:
                for t in s2:
                    counter[self.wi[h], self.wi[t]] += cnt

        def symmetric_count(s1, s2, cnt, counter):
            for w1 in s1:
                for w2 in s2:
                    counter[self.order(self.wi[w1], self.wi[w2])] += cnt

        head_count = Counter()
        tail_count = Counter()
        head_head_count = Counter()
        tail_tail_count = Counter()
        head_tail_count = Counter()
        tail_head_count = Counter()
        for le, cnt in tqdm(zip(lemma, count)):
            head_words, tail_words = le
            for w in head_words:
                head_count[self.wi[w]] += cnt
            for w in tail_words:
                tail_count[self.wi[w]] += cnt
                symmetric_count(head_words, head_words, cnt, head_head_count)
                symmetric_count(tail_words, tail_words, cnt, tail_tail_count)
                asymmetric_count(head_words, tail_words, cnt, head_tail_count)
                asymmetric_count(tail_words, head_words, cnt, tail_head_count)

        ind_freq_counters = [head_count, tail_count]
        co_freq_counters = [
            head_head_count, tail_tail_count, head_tail_count, tail_head_count
        ]
        return ind_freq_counters, co_freq_counters

    def ixf(self, data, index):
        """Filter data using index"""
        return [data[i] for i in index]

    def count_occurrences(self, clear=True):
        """Count word (co)occurrences w.r.t. their partitions"""
        if clear:
            self.ind_freq = {}
            self.co_freq = {}
            self.freq_total_by_partition = {}
            self.pmi = {}
            self.partition_pmi = {}
            self.ind_freq_total = 0
            self.co_freq_total = 0
            self.co_freq_sum = {}
            self.co_pmi = {}

        # count individual and coocurrence frequency by each partition
        for pname, partition in tqdm(self.partitions.items()):
            _ind_freq = {}
            _co_freq = {}
            _freq_total_by_partition = {}
            _pmi = {}
            ind_freq_counters, co_freq_counters = self.count_freq(
                self.ixf(self.lemma, partition['ix']),
                self.ixf(self.count, partition['ix']))

            # convert frequency count to sparse matrix and calculate pmi
            # individual frequency
            for t, counter in zip(self.ind_freq_types, ind_freq_counters):
                freq_sum = sum(list(counter.values()))
                _freq_total_by_partition[t] = freq_sum
                _ind_freq[t] = counter
                self.ind_freq_total += freq_sum

            # cooccurrence frequency
            for t, counter in zip(self.co_freq_types, co_freq_counters):
                freq_sum = sum(list(counter.values()))
                _freq_total_by_partition[t] = freq_sum
                sparse_matrix = self.counter_to_sparse_matrix(counter)
                _co_freq[t] = sparse_matrix
                pmi_matrix = self.calc_pmi(sparse_matrix, 1)
                _pmi[t] = pmi_matrix
                self.co_freq_total += freq_sum

            self.freq_total_by_partition[pname] = _freq_total_by_partition
            self.ind_freq[pname] = _ind_freq
            self.co_freq[pname] = _co_freq
            self.pmi[pname] = _pmi

        # count individual and coocurrence frequency by each frequency types
        for t in self.co_freq_types:
            co_freqs = [cf[t] for cf in self.co_freq.values()]
            _co_freq_sum = self._sum_sparse_matrix(co_freqs)
            self.co_freq_sum[t] = _co_freq_sum
            pmi_matrix = self.calc_pmi(_co_freq_sum, 1)
            self.co_pmi[t] = pmi_matrix

        all_co_freq = [cf for cf in self.co_freq_sum.values()]
        self.all_co_freq_sum = self._sum_sparse_matrix(all_co_freq)
        self.all_pmi = self.calc_pmi(self.all_co_freq_sum, 1)

    def _sum_sparse_matrix(self, matrixs):
        matrix_sum = matrixs[0]
        for m in matrixs[1:]:
            matrix_sum += m
        return matrix_sum

    def counter_to_sparse_matrix(self, counter):
        row = []
        col = []
        data = []
        for record in counter.items():
            (r, c), d = record
            row.append(r)
            col.append(c)
            data.append(d)
        sparse_matrix = csr_matrix(
            (data, (row, col)), shape=(self.vocab_size, self.vocab_size))
        return sparse_matrix

    def calc_pmi(self, counts, cds=1):
        """calc_pmi
        We have reused the pmi calculation code from:
        https://bitbucket.org/omerlevy/hyperwords

        Please cite:
            "Improving Distributional Similarity
            with Lessons Learned from Word Embeddings"
            Omer Levy, Yoav Goldberg, and Ido Dagan. TACL 2015.

        :param counts: csr_matrix containing cooccur count
        :param cds: Context distribution smoothing
        """

        def safe_reciprocal(x, default=0):
            return 1 / x if x != 0 else default

        sum_w = np.array(counts.sum(axis=1))[:, 0]
        sum_c = np.array(counts.sum(axis=0))[0, :]
        if cds != 1:
            sum_c = sum_c**cds
        sum_total = sum_c.sum()
        # safe reciprocal: return 0 for x/0
        sum_w = np.array(list(map(safe_reciprocal, sum_w)))
        sum_c = np.array(list(map(safe_reciprocal, sum_c)))

        pmi = csr_matrix(counts)
        pmi = self._multiply_by_rows(pmi, sum_w)
        pmi = self._multiply_by_columns(pmi, sum_c)
        pmi = pmi * sum_total
        return pmi

    def _multiply_by_rows(self, matrix, row_coefs):
        normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
        normalizer.setdiag(row_coefs)
        return normalizer.tocsr().dot(matrix)

    def _multiply_by_columns(self, matrix, col_coefs):
        normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
        normalizer.setdiag(col_coefs)
        return matrix.dot(normalizer.tocsr())

    def statistical_features(self, word1, word2):
        """statistical_features

        Extract statistical features for word1 and word2

        features includes:
            - Word frequency
            - Cooccurrence frequency
            - (Asymmetric) Pointwise Mutual Information

        :param word1: word1(symmetric) / head(asymmetric)
        :param word2: word2(symmetric) / tail(asymmetric)
        """
        w1 = self.wi.get(word1, 0)
        w2 = self.wi.get(word2, 0)

        heads = []
        tails = []
        co_occurs = {t: [] for t in self.co_freq_types}
        for p in self.partition_types:
            # head word frequency
            heads.append(self.ind_freq[p]['head'][w1])
            yield self.ind_freq[p]['head'][w1]
            # tail word frequency
            tails.append(self.ind_freq[p]['tail'][w2])
            yield self.ind_freq[p]['tail'][w2]

            for t in self.co_freq_types:
                c1, c2 = w1, w2
                if t in self.sym_co_freq_types:
                    c1, c2 = self.order(w1, w2)
                co_occurs[t].append(self.co_freq[p][t][c1, c2])
                yield self.co_freq[p][t][c1, c2]
                yield self.pmi[p][t][c1, c2]

        all_heads_count = sum(heads)
        all_tails_count = sum(tails)
        all_co_occur_count = {t: sum(co_occurs[t]) for t in self.co_freq_types}
        yield all_heads_count
        yield all_tails_count
        for t in self.co_freq_types:
            c1, c2 = w1, w2
            if t in self.sym_co_freq_types:
                c1, c2 = self.order(w1, w2)
            yield all_co_occur_count[t]
            yield self.co_pmi[t][c1, c2]

        # totalpmi(x, y)
        yield self.all_pmi[c1, c2]

    def save_model(self, save_dir):
        """save_model

        Save statistical features, i.e. frequencies, apmi, with pickle.
        Attributes in self.ignore_attributes are not saved.

        :param save_dir: directory for saving pickles
        """
        os.makedirs(self.config.model_dir)
        for attribute in list(self.__dict__.keys()):
            if attribute not in self.ignore_attributes:
                save_path = os.path.join(save_dir, attribute + '.pickle')
                with open(save_path, 'wb') as handle:
                    pickle.dump(
                        getattr(self, attribute),
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, load_dir):
        """load_model

        Load attributes from pickles in load_dir

        :param load_dir: directory for loading pickles
        """
        attribute_pickles = os.listdir(load_dir)
        for attribute in list(attribute_pickles):
            with open(os.path.join(load_dir, attribute), 'rb') as handle:
                setattr(self, attribute.split('.')[0], pickle.load(handle))
