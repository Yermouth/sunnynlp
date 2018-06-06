import spacy


class NLP(object):
    def __init__(self, language):
        self.nlp = spacy.load(language)

    def lemma(self, word):
        doc = self.nlp(word)
        return doc[0].lemma_
