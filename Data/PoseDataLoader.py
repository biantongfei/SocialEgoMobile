from torch.utils.data import DataLoader
import torch
from torch_geometric.data import Data, Batch

import numpy as np
import random

from constants import coco_body_point_num, coco_body_l_pair, device, dtype

body_edge_index = torch.Tensor(coco_body_l_pair).t().to(dtype=torch.int64, device=device)
body_l_pair_num = len(coco_body_l_pair)


class Pose_DataLoader(DataLoader):
    def __init__(self, dataset, batch_size, sequence_length, keypoint_drop_rate, frame_drop_rate):
        super().__init__(dataset=dataset, batch_size=batch_size, num_workers=0, collate_fn=self.gcn_collate_fn)
        self.sequence_length = sequence_length
        self.keypoint_drop_rate = keypoint_drop_rate
        self.frame_drop_rate = frame_drop_rate

    def gcn_collate_fn(self, data):
        body_pose_graph_data = []
        int_label = []
        att_label = []
        act_label = []
        for d in data:
            for i in range(self.sequence_length):
                frame = d[0][i]
                frame = self.drop_keypoints(frame)
                frame = self.drop_frames(frame)
                body_pose_graph_data.append(
                    Data(x=frame[:coco_body_point_num].to(dtype=dtype, device=device), edge_index=body_edge_index))
            int_label.append(d[1][0])
            att_label.append(d[1][1])
            act_label.append(d[1][2])
        labels = (torch.Tensor(int_label, device=device),
                  torch.Tensor(att_label, device=device),
                  torch.Tensor(act_label, device=device))
        return Batch.from_data_list(body_pose_graph_data), labels

    def drop_keypoints(self, pose_sequence):
        mask = np.ones((1, pose_sequence.shape[1], 1), dtype=np.uint8)
        num_zeros = int(self.keypoint_drop_rate * pose_sequence.shape[1])
        zero_indices = np.random.choice(pose_sequence.shape[1], num_zeros, replace=False)
        mask.flat[zero_indices] = 0
        mask = torch.tensor(mask).expand(self.sequence_length, pose_sequence.shape[1], 3)
        if random.random() < 0.5:
            pose_sequence = pose_sequence * mask
        else:
            noise = torch.rand_like(pose_sequence[:, :, :2]) - 0.5
            noise = torch.cat([noise, torch.zeros(pose_sequence.shape[0], pose_sequence.shape[1], 1)], dim=2)
            pose_sequence = pose_sequence * mask + noise * (1 - mask)
        return pose_sequence

    def drop_frames(self, pose_sequence):
        num_zeros = int(self.frame_drop_rate * self.sequence_length)
        zero_indices = np.random.choice(self.sequence_length, num_zeros, replace=False)
        if random.random() < 0.5:
            mask = np.ones((self.sequence_length, 1, 1), dtype=np.uint8)
            mask.flat[zero_indices] = 0
            mask = torch.tensor(mask).expand(self.sequence_length, pose_sequence.shape[1], 3)
            noise = torch.rand_like(pose_sequence[:, :, :2]) - 0.5
            noise = torch.cat([noise, torch.zeros(pose_sequence.shape[0], pose_sequence.shape[1], 1)], dim=2)
            pose_sequence = pose_sequence * mask + noise * (1 - mask)
        else:
            zero_indices.sort()
            for index in zero_indices:
                if index == 0:
                    pose_sequence[index] = torch.zeros((pose_sequence.shape[1], 3))
                else:
                    pose_sequence[index] = pose_sequence[index - 1]
        return pose_sequence
