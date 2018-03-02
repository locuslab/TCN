import os
import torch
from torch.autograd import Variable
import re
from collections import Counter
import pickle

"""
Note: The meaning of batch_size in PTB is different from that in MNIST example. In MNIST, 
batch_size is the # of sample data that is considered in each iteration; in PTB, however,
it is the number of segments to speed up computation. 

The goal of PTB is to train a language model to predict the next word.
"""

def data_generator(args):
    if os.path.exists(args.data + "/corpus") and not args.corpus:
        corpus = pickle.load(open(args.data + '/corpus', 'rb'))
    else:
        print("Creating Corpus...")
        corpus = Corpus(args.data + "/lambada_vocabulary_sorted.txt", args.data)
        pickle.dump(corpus, open(args.data + '/corpus', 'wb'))

    eval_batch_size = 1
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = [[0] * (args.seq_len-len(line)) + line for line in corpus.valid]
    test_data = [[0] * (args.seq_len-len(line)) + line for line in corpus.test]
    return train_data, val_data, test_data, corpus


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, dict_path, path):
        self.dictionary = Dictionary()
        self.prep_dict(dict_path)
        self.train = torch.LongTensor(self.tokenize(os.path.join(path, 'train-novels')))
        self.valid = self.tokenize(os.path.join(path, 'lambada_development_plain_text.txt'), eval=True)
        self.test = self.tokenize(os.path.join(path, 'lambada_test_plain_text.txt'), eval=True)

    def prep_dict(self, dict_path):
        assert os.path.exists(dict_path)

        # Add words to the dictionary
        with open(dict_path, 'r') as f:
            tokens = 0
            for line in f:
                word = line.strip()
                tokens += 1
                self.dictionary.add_word(word)

        if "<eos>" not in self.dictionary.word2idx:
            self.dictionary.add_word("<eos>")
            tokens += 1

        print("The dictionary captured a vocabulary of size {0}.".format(tokens))

    def tokenize(self, path, eval=False):
        assert os.path.exists(path)

        ids = []
        token = 0
        misses = 0
        if not path.endswith(".txt"):   # it's a folder
            for subdir in os.listdir(path):
                for filename in os.listdir(path + "/" + subdir):
                    if filename.endswith(".txt"):
                        full_path = "{0}/{1}/{2}".format(path, subdir, filename)
                        # Tokenize file content
                        delta_ids, delta_token, delta_miss = self._tokenize_file(full_path, eval=eval)
                        ids += delta_ids
                        token += delta_token
                        misses += delta_miss
        else:
            ids, token, misses = self._tokenize_file(path, eval=eval)

        print(token, misses)
        return ids

    def _tokenize_file(self, path, eval=False):
        with open(path, 'r') as f:
            token = 0
            ids = []
            misses = 0
            for line in f:
                line_ids = []
                words = line.strip().split() + ['<eos>']
                if eval:
                    words = words[:-1]
                for word in words:
                    # These words are in the text but not vocabulary
                    if word == "n't":
                        word = "not"
                    elif word == "'s":
                        word = "is"
                    elif word == "'re":
                        word = "are"
                    elif word == "'ve":
                        word = "have"
                    elif word == "wo":
                        word = "will"
                    if word not in self.dictionary.word2idx:
                        word = re.sub(r'[^\w\s]', '', word)
                    if word not in self.dictionary.word2idx:
                        misses += 1
                        continue
                    line_ids.append(self.dictionary.word2idx[word])
                    token += 1
                if eval:
                    ids.append(line_ids)
                else:
                    ids += line_ids
        return ids, token, misses


def batchify(data, batch_size, args):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    print(data.size())
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.seq_len, source.size(1) - 1 - i)
    data = Variable(source[:, i:i+seq_len], volatile=evaluation)
    target = Variable(source[:, i+1:i+1+seq_len])  # CAUTION: This is un-flattened!
    return data, target
