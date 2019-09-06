from tqdm import tqdm
import torch
import torch.optim as optim
from CalPerformance import cal_performance
import math
import time
from DataSet import MyDataSet
from Transformer import Transformer
from MyOptim import MyOptimizer
from LoadData import DataLoader
import numpy as np
PAD = 0


def train_epoch(model, training_data, optimizer, indexdict, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    PAD = indexdict['pad']
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq = torch.Tensor(np.array(batch.src))
        tgt_seq = torch.Tensor(np.array(batch.trg))
        #print(src_seq.size())
        src_seq = src_seq.transpose(0, 1).to(device)
        tgt_seq = tgt_seq.transpose(0, 1).to(device)
        src_len = torch.zeros(src_seq.size(0), ).fill_(src_seq.size(1)).long().to(device)
        tgt_len = torch.zeros(tgt_seq.size(0), ).fill_(tgt_seq.size(1)).long().to(device)
        tgt_seq_y = tgt_seq[:, :-1]
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred, enc_self_attn, dec_self_attn, ctx_attn = model(src_seq, src_len, tgt_seq_y, tgt_len-1)
        #print(pred.size())
        #print(gold.size())
        # backward
        loss, n_correct = cal_performance(pred, gold, PAD, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, indexdict,device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    PAD = indexdict['pad']
    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq = torch.Tensor(np.array(batch.src))
            tgt_seq = torch.Tensor(np.array(batch.trg))
            # print(src_seq.size())
            src_seq = src_seq.transpose(0, 1).to(device)
            tgt_seq = tgt_seq.transpose(0, 1).to(device)
            src_len = torch.zeros(src_seq.size(0), ).fill_(src_seq.size(1)).long()
            tgt_len = torch.zeros(tgt_seq.size(0), ).fill_(tgt_seq.size(1)).long()
            tgt_seq_y = tgt_seq[:, :-1]
            gold = tgt_seq[:, 1:]

            # forward
            pred, enc_self_attn, dec_self_attn, ctx_attn = model(src_seq, src_len, tgt_seq_y, tgt_len - 1)

            loss, n_correct = cal_performance(pred, gold, PAD, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, validation_data, optimizer, indexdict, device, epoch):
    ''' Start training '''

    valid_accus = []
    for epoch_i in range(epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, indexdict, device, smoothing=True)
        print('  - (Training)    accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, indexdict, device)
        print('  - (Validation)  accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    src_voc_size = 11
    tgt_voc_size = 11
    train_loader=torch.utils.data.DataLoader(
        MyDataSet(V=11, batch=20),
        num_workers=2,
        batch_size=30,
    )
    valid_loader = torch.utils.data.DataLoader(
        MyDataSet(V=11, batch=20),
        num_workers=2,
        batch_size=30,
    )
    '''
    traindata, valdata, testdata, indexdict, (src_maxlen, trg_maxlen), (src_voc_size, tgt_voc_size) = \
        DataLoader(32).get_data_iterator()
    model = Transformer(src_vocab_size=src_voc_size,
               src_max_len=src_maxlen,
               tgt_vocab_size=tgt_voc_size,
               tgt_max_len=trg_maxlen)

    optimizer = MyOptimizer(model_size=512, factor=1.0, warmup=400,
                          optimizer=optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09))
    train(model, traindata, valdata, optimizer, indexdict, device, 25)

if __name__ == '__main__':
    main()