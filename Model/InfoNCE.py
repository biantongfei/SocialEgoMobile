import torch.nn as nn
import torch.nn.functional as F
import torch

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, student_feat, teacher_feat):
        # 计算相似度矩阵 (cosine similarity)
        similarity_matrix = F.cosine_similarity(student_feat.unsqueeze(1), teacher_feat.unsqueeze(0), dim=2)

        # 取对角线作为正样本的相似度
        positive_sim = torch.diag(similarity_matrix)

        # 计算 InfoNCE 损失
        numerator = torch.exp(positive_sim / self.temperature)
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature), dim=1)
        loss = -torch.log(numerator / denominator).mean()
        return loss