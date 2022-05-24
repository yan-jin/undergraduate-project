import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy


class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
    """

    def __init__(self, encoder, config):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, 17)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source_ids=None, source_mask=None, source_labels=None):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        # encoder_output: (batch_size, hidden_size)
        encoder_output = outputs['pooler_output'].contiguous()
        out = self.dense(encoder_output)
        out = self.softmax(out)

        if source_labels is not None:
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(out, source_labels)
            return out, loss
        else:
            return torch.argmax(out)
