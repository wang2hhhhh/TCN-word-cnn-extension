import os
import torch
import pickle
from transformers import BertTokenizer, DistilBertTokenizer

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
        corpus = Corpus(args.data)
        pickle.dump(corpus, open(args.data + '/corpus', 'wb'))
    return corpus

class Corpus(object):
    def __init__(self, path, bertmodel = "distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(bertmodel)
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))


    def tokenize(self, path):
        assert os.path.exists(path)
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.tensor([])
            for line in f:
                ids = torch.concat((ids, self.tokenizer(line, return_tensors = 'pt')['input_ids'].squeeze(dim = 0)))
        return ids.to(torch.int64)

def batchify(data, batch_size, args):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.seq_len, source.size(1) - 1 - i)

    if evaluation == True:
      with torch.no_grad():
        data = source[:, i:i+seq_len]
    else:
      data = source[:, i:i+seq_len]
    target = source[:, i+1:i+1+seq_len]    # CAUTION: This is un-flattened!
      
    return data, target
