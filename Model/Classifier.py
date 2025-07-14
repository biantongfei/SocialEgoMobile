from torch import nn
import torch

from constants import intention_classes, attitude_classes, jpl_action_classes, harper_action_classes

intention_class_num = len(intention_classes)
attitude_class_num = len(attitude_classes)
jpl_action_class_num = len(jpl_action_classes)
harper_action_class_num = len(harper_action_classes)


class Chain_Classifier(nn.Module):
    def __init__(self, dataset, in_feature_size=16):
        super(Chain_Classifier, self).__init__()
        super().__init__()
        action_class_num = jpl_action_class_num if dataset == 'JPL' else harper_action_class_num
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(in_feature_size, intention_class_num)
                                            )

        self.attitude_head = nn.Sequential(
            nn.BatchNorm1d(in_feature_size + intention_class_num),
            nn.ReLU(),
            nn.Linear(in_feature_size + intention_class_num, attitude_class_num)
        )
        self.action_head = nn.Sequential(
            nn.BatchNorm1d(in_feature_size + intention_class_num + attitude_class_num),
            nn.ReLU(),
            nn.Linear(in_feature_size + intention_class_num + attitude_class_num, action_class_num)
        )

    def forward(self, y):
        y1 = self.intention_head(y)
        y2 = self.attitude_head(torch.cat((y, y1), dim=1))
        y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
        return y1, y2, y3
