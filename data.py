import os
import torch
import pickle
from collections import Counter
import numpy as np
test_mode=False

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
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            if id_path is not None:
                f_uid = open(id_path,'r', encoding='utf-8')
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

class RefinedCorpus(object):
    # A refined corpus class that treats each sentence separately and carefully handle each sentence
    def __init__(self, path, dictionary, bsz, args):
        self.dictionary = dictionary
        self.train, self.train_uids, self.train_dict = self.tokenize(os.path.join(path, 'train.txt'), bsz, args)
        self.valid, self.valid_uids, self.valid_dict = self.tokenize(os.path.join(path, 'valid.txt'), bsz, args)
        self.test, self.test_uids, self.test_dict = self.tokenize(os.path.join(path, 'test.txt'), bsz, args)

        
    def tokenize(self, path, bsz, args):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        id_path = path.replace('.txt','_id.txt')

        with open(path, 'r', encoding='utf-8') as f:
            if id_path is not None:
                f_uid = open(id_path,'r', encoding='utf-8')
            all_lines = [line for line in f]
            all_uids = [uid.strip() for uid in f_uid] 
            if test_mode:
                all_lines=all_lines[:500]
                all_uids=all_uids[:500]
                print('using first 500 sents.')

        burnin_lines = all_lines[:args.burn_in]

        tokens = 0
        for line in all_lines:
            words = line.split() + ['<eos>']
            tokens += len(words)
        
        nbatch_est = tokens//bsz
        ids = [[] for idx in range(bsz)]
        uids = [[] for idx in range(bsz)]
        sent_dict={}
        # Tokenize file content



        for lid, line in enumerate(burnin_lines):
            words = line.split() + ['<eos>']
            for word in words:
                for bidx in range(bsz):
                    ids[bidx].append(self.word2idx(word))
                    uids[bidx].append('burnin_{}_{}'.format(lid,bidx))
            sent_dict['burnin_{}_{}'.format(lid,bidx)]=line
        nbatch_est+=len(ids[0])

        current_b_line = 0
        for line,uid in zip(all_lines, all_uids):
            words = line.split() + ['<eos>']
            for word in words:
                ids[current_b_line].append(self.word2idx(word))
                uids[current_b_line].append(uid)
            if len(ids[current_b_line]) >= nbatch_est:
                current_b_line += 1
                assert current_b_line < bsz
            sent_dict[uid]=line
        overall_length = max([len(item) for item in ids])
        for words,uid in zip(ids,uids):
            while len(words)<overall_length:
                words.append(self.word2idx('<eos>'))
                uid.append('pad')
        ids = torch.LongTensor(ids).t().contiguous()
        uids = np.array(uids).transpose()
        if args.cuda:
            ids = ids.cuda()

        return ids, uids, sent_dict
    def word2idx(self, word):
        try:
            idx=self.dictionary.word2idx[word]
        except:
            idx = self.dictionary.word2idx['<unk>']
        return idx



