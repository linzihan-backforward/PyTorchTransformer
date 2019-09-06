import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

def padding_mask(seq_k, seq_q):
    """

    :param seq_k: key 序列
    :param seq_q: query 序列
    :return: Attention 中用到的mask矩阵 size与 Q.dot(K^T) 相同
    """
	# seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

class Transformer(nn.Module):

    def __init__(self,
               src_vocab_size,
               src_max_len,
               tgt_vocab_size,
               tgt_max_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)

        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        nn.init.xavier_normal_(self.linear.weight)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):

        context_attn_mask = padding_mask(src_seq, tgt_seq)

        output, enc_self_attn = self.encoder(src_seq, src_len)

        output, dec_self_attn, ctx_attn = self.decoder(
          tgt_seq, tgt_len, output, context_attn_mask)

        output = self.linear(output)
        output = self.softmax(output)
        output = output.view(-1, output.size(2))

        return output, enc_self_attn, dec_self_attn, ctx_attn
