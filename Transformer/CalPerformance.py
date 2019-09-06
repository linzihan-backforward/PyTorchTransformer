import torch
import torch.nn.functional as F


def cal_performance(pred, gold, PAD, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, PAD, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(PAD)
    n_correct = pred.eq(gold.long())
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct

def cal_loss(pred, gold, PAD,smoothing):
    '''

    Calculate cross entropy loss, apply label smoothing if needed.
    '''

    gold = gold.contiguous().view(-1)
    # 将label转换为[batch*l,1] 再转成onehot
    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1).long(), 1)
    pred = torch.log(pred)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    non_pad_mask = gold.ne(PAD)
    loss = -(one_hot * pred).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).sum()  # average later
    return loss
