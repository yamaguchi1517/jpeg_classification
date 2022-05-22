from logging import getLogger
from typing import Callable, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


logger = getLogger('models.classification_lstm')

#nn.Moduleの継承
class Classification_lstm(nn.Module):
    def __init__(
        self, 
        embedding_dim,
        hidden_dim, 
        num_classes: int = 0
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        #ノイズによって単語IDが増えるため、350に設定
        self.embedding = nn.Embedding(350, embedding_dim, padding_idx=0)
        logger.info('単語埋め込み層を生成しました。')
        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_dim,
                            batch_first = True)
        logger.info('LSTM層を生成しました。')
        self.dense = nn.Linear(hidden_dim, num_classes)
        logger.info('全結合層を定義しました。')
        
    def forward(self, input):
        emb = self.embedding(input)
        _, lstm_out = self.lstm(emb)
        output = self.dense(lstm_out[0].view(-1, self.hidden_dim))

        return output