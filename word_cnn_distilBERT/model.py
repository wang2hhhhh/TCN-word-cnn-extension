import torch
from torch import nn
import sys
sys.path.append("../../")
from TCN.tcn import TemporalConvNet
from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer


class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, bertmodel = "distilbert-base-uncased" ):
        super(TCN, self).__init__()
        self.encoder = DistilBertModel.from_pretrained(bertmodel)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], output_size)

        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout

    def forward(self, input):

        emb = self.drop(self.encoder(input).last_hidden_state)
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous()


