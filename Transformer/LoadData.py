import torch
from torchtext.datasets import  Multi30k
from torchtext.data import Field, BucketIterator
import spacy


class DataLoader:
    def __init__(self, batchsize = 64):
        """
        load Multi30k dataSet, return train/valid/test dataIterator and tokenIndex
        :param batchsize:
        """
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        SRC = Field(
            tokenize=tokenize_de,
            init_token='<bos>',
            eos_token='<eos>',
            lower=True
        )
        TRG = Field(
            tokenize=tokenize_en,
            init_token='<bos>',
            eos_token='<eos>',
            lower=True
        )

        train_data, valid_data, test_data = Multi30k.splits(exts=('.de','.en'),fields=(SRC,TRG))
        src_maxlen=0
        trg_maxlen=0
        for example in train_data:
            e = vars(example)
            src_maxlen = max(len(e['src']), src_maxlen)
            trg_maxlen = max(len(e['trg']), trg_maxlen)

        for example in valid_data:
            e = vars(example)
            src_maxlen = max(len(e['src']), src_maxlen)
            trg_maxlen = max(len(e['trg']), trg_maxlen)

        self.src_maxlen = src_maxlen
        self.trg_maxlen = trg_maxlen

        #print(src_maxlen, trg_maxlen)
        SRC.build_vocab(train_data, min_freq=2)
        TRG.build_vocab(train_data, min_freq=2)

        self.srcvoc = len(SRC.vocab)
        self.trgvoc = len(TRG.vocab)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=batchsize,
            device=dev)
        UNk_IDX = TRG.vocab.stoi['<unk>']
        PAD_IDX = SRC.vocab.stoi['<pad>']
        BOS_IDX = SRC.vocab.stoi['<bos>']
        EOS_IDX = SRC.vocab.stoi['<eos>']
        self.IndexDict = {'unk': UNk_IDX, 'pad': PAD_IDX, 'bos': BOS_IDX, 'eos': EOS_IDX, }

    def get_data_iterator(self):
        return self.train_iterator, self.valid_iterator, self.test_iterator, self.IndexDict, \
               (self.src_maxlen, self.trg_maxlen), (self.srcvoc, self.trgvoc)
#data = DataLoader(32)
