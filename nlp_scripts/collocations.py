import nltk
import math
import itertools
from nltk.util import ngrams as ng
from collections import defaultdict

"""
Used for filtering phrases.
"""
ADJ_TAGS = ['JJ', 'JJR', 'JJS']
NOUN_TAGS = ['NN', 'NNP', 'NNPS', 'NNS']
PUNCTUATION = "!?\".-,:;"
BIGRAM_TAG_PATTERNS = list(itertools.product(
    ADJ_TAGS+NOUN_TAGS,
    NOUN_TAGS
))
TRIGRAM_TAG_PATTERNS = list(itertools.product(ADJ_TAGS, ADJ_TAGS, NOUN_TAGS)) + \
    list(itertools.product(ADJ_TAGS, NOUN_TAGS, NOUN_TAGS)) + \
    list(itertools.product(NOUN_TAGS, ADJ_TAGS, NOUN_TAGS)) + \
    list(itertools.product(NOUN_TAGS, NOUN_TAGS, NOUN_TAGS)) + \
    list(itertools.product(NOUN_TAGS, ['IN'], NOUN_TAGS))


class Collocation(object):
    def __init__(self, text):
        if str(text) == text:
            self.text = nltk.word_tokenize(text)
        else:
            self.text = text
        self.collocation_dict = {}

    def t_squared_collocations(self, n=2, top=100):
        ngrams = ng(self.text, n)
        ngram_freqs = nltk.FreqDist(list(ngrams))
        ngram_count = len(ngram_freqs)
        frequencies = nltk.FreqDist(self.text)
        collocation_dict = defaultdict(int)
        for ngram, freq in ngram_freqs.items():
            t_stat = Collocation.t_statistic(
                freq/ngram_count,
                frequencies[ngram[0]]/ngram_count * frequencies[ngram[1]]/ngram_count,
                freq/ngram_count,
                ngram_count
            )
            if t_stat < 2.576 or not Collocation.valid_collocation(nltk.pos_tag(ngram), n):
                continue
            collocation = ''
            for i in range(n):
                collocation += ngram[i] + ' '
            collocation_dict[collocation[:-1]] = t_stat
        self.collocation_dict[n] = self.sort_dict(collocation_dict, top)

    def frequency_collocations(self, n=2, top=100):
        """A simple implementation for finding n-word long collocations

        Arguments:
            n (int): The length of collocations to search for.
            top (int): Specify the number of collocations returned, e.g.,
                top=100 returns the 100 most common collocations.
        """
        collocation_dict = defaultdict(int)
        for ngram in ng(self.text, n):
            tagged = nltk.pos_tag(ngram)
            if Collocation.valid_collocation(tagged, n):
                collocation = ''
                for i in range(n):
                    collocation += tagged[i][0] + ' '
                collocation_dict[collocation[:-1]] += 1

        self.collocation_dict[n] = self.sort_dict(collocation_dict, top)

    @staticmethod
    def sort_dict(dictionary, top):
        sorted_dictionary = sorted(dictionary.items(), key=lambda x: x[1])
        sorted_dictionary.reverse()
        final_list = []
        for i in range(top):
            try:
                final_list.append(sorted_dictionary[i])
            except IndexError:
                return final_list
        return final_list

    @staticmethod
    def overlap(str1, str2):
        for character in str1:
            if character in str2:
                return True
        return False

    @staticmethod
    def valid_collocation(candidate, n):
        """TODO

        Arguments:
            candidate ([]):
            n (int):

        Return:
            Boolean: Represents whether the collocation is valid or not.
        """
        for i in range(n):
            if Collocation.overlap(candidate[i][0], PUNCTUATION):
                return False
        for i in range(n):
            if candidate[i][0] == 's':
                return False
        if n == 2:
            valid_pattern = (candidate[0][1],
                             candidate[1][1]) in BIGRAM_TAG_PATTERNS
        elif n == 3:
            valid_pattern = (candidate[0][1],
                             candidate[1][1],
                             candidate[2][1]) in TRIGRAM_TAG_PATTERNS
        else:
            raise ValueError('We only support 2 and 3 word long collocations')
        return valid_pattern

    @staticmethod
    def t_statistic(sample_mean, dist_mean, sample_variance, sample_size):
        num = sample_mean - dist_mean
        denom = math.sqrt(sample_variance/sample_size)
        return num/denom


# Exercise 5.10
def distinguishing_terms(corpus, split):
    """Terribly inefficient way to find distinguishing bigrams across some
    split of a corpus.

    Arguments:
        corpus ([string]): A tokenized text corpus.
        split (int): Where to divide the text.

    Returns:
        [string], [string]: Two lists corresponding to the distinguishing
            bigrams in the first and second portions of the corpus.
    """
    # Get collocation counts for the first and second portions of corpus.
    h1 = Collocation(list(corpus[:split]))
    h1.frequency_collocations()
    h1dict = dict(h1.collocation_dict[2])
    h2 = Collocation(list(corpus[split:]))
    h2.frequency_collocations()
    h2dict = dict(h2.collocation_dict[2])
    first_half = []
    snd_half = []
    for key, val in h1dict.items():
        if key in h2dict:
            if val/(val + h2dict[key]) > 0.8:
                first_half.append(key)
        else:
            first_half.append(key)
    for key, val in h2dict.items():
        if key in h1dict:
            if val/(val + h1dict[key]) > 0.8:
                snd_half.append(key)
        else:
            snd_half.append(key)
    return first_half, snd_half
