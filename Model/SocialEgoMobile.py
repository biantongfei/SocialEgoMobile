from torch import nn
from torch_geometric.nn import GAT
import torch

from Model.Classifier import Chain_Classifier
from constants import coco_body_point_num


class SocialEgoMobile(nn.Module):
    def __init__(self, sequence_length, gcn_layers, lstm_layers, keypoint_hidden_dim, lstm_hidden_dim,
                 fc_hidden1, fc_hidden2, dataset):
        super(SocialEgoMobile, self).__init__()
        self.sequence_length = sequence_length
        self.keypoint_hidden_dim = keypoint_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        self.body_point_num = coco_body_point_num
        self.GAT_body = GAT(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=gcn_layers)
        self.gcn_attention = nn.Linear(self.keypoint_hidden_dim * self.body_point_num, 1)
        self.lstm = nn.LSTM(self.body_point_num * self.keypoint_hidden_dim, hidden_size=self.lstm_hidden_dim,
                            num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.fc_input_size = self.lstm_hidden_dim * 2
        self.lstm_attention = nn.Linear(self.fc_input_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_hidden1),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_hidden2),
        )
        self.classifier = Chain_Classifier(dataset, self.fc_hidden2)

    def forward(self, data, representation=False):
        x_body = self.GCN_body(x=data.x, edge_index=data.edge_index, batch=data.batch)
        x = x_body.view(-1, self.sequence_length, self.keypoint_hidden_dim * self.body_point_num)
        gcn_attention_weights = nn.Softmax(dim=1)(self.gcn_attention(x))
        x = x * gcn_attention_weights

        on, _ = self.lstm(x)
        on = on.view(on.shape[0], on.shape[1], 2, -1)
        x = (torch.cat([on[:, :, 0, :], on[:, :, 1, :]], dim=-1))
        attention_weights = nn.Softmax(dim=1)(self.lstm_attention(x))
        x = torch.sum(x * attention_weights, dim=1)

        x = self.fc(x)
        return (self.classifier(x), x) if representation else self.classifier(x)
