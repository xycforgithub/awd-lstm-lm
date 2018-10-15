import os
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, get_id=False):
        self.dictionary = Dictionary()
        self.train, self.train_uids = self.tokenize(os.path.join(path, 'train.txt'),get_id)
        self.valid, self.valid_uids = self.tokenize(os.path.join(path, 'valid.txt'),get_id)
        self.test, self.test_uids = self.tokenize(os.path.join(path, 'test.txt'),get_id)

    def tokenize(self, path, get_id=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        id_path = path.replace('.txt','_id.txt') if get_id else None
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            if id_path is not None:
                f_uid = open(id_path,'r')
            ids = torch.LongTensor(tokens)
            uids = []
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                uid = f_uid.readline().strip() if id_path is not None else None
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
                    uids.append(uid)


        return ids, uids
