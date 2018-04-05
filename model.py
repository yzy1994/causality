import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class SiameseNetwork(nn.Module,):
    def __init__(self, word_nums, embedding_dims, hidden_dims, pretrained_embedding, use_gru, dropout_rate, alpha):
        super(SiameseNetwork, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=word_nums , embedding_dim=embedding_dims, padding_idx=0)
        self.embedding_layer.weight.data.copy_(pretrained_embedding)

        if use_gru:
            self.rnn = nn.GRU(input_size=embedding_dims, hidden_size=hidden_dims, bidirectional=True, dropout=dropout_rate)
        else:
            self.rnn = nn.LSTM(input_size=embedding_dims, hidden_size=hidden_dims, bidirectional=True, dropout=dropout_rate)


        self.linear = nn.Linear(hidden_dims * 2, 3)
        self.alpha = alpha

    def forward_once(self, x, x_e):
        x_e = x_e.float()
        out , _ = self.rnn(x)
        out1 = out[:, 5, :]
        out2 = out[:, :, :]

        out2 = torch.bmm(x_e, out2)

        out2 = torch.squeeze(out2, 1)

        output = (1 - self.alpha) * out1 + self.alpha * out2
        return output

    def forward(self, x1, x2, x1_e, x2_e):
        x1 = x1.long()
        x2 = x2.long()
        x1 = self.embedding_layer(x1)
        x2 = self.embedding_layer(x2)
        out1 = self.forward_once(x1, x1_e)
        out2 = self.forward_once(x2, x2_e)
        hidden = out1 - out2

        result = self.linear(hidden)
        result = F.softmax(result)
        return result

    def generate_event_embedding(self, x, x_e):
        x = self.embedding_layer(x)
        out = self.forward_once(x, x_e)
        return out