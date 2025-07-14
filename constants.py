import os
import torch
import numpy as np
import random


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if torch.cuda.is_available():
    print('Using CUDA for training')
    device = torch.device("cuda:0")
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # device = torch.device('cpu')
else:
    print('Using CPU for training')
    device = torch.device('cpu')
dtype = torch.float32

coco_body_point_num = 17
head_point_num = 68
hands_point_num = 42

intention_classes = ['Interacting', 'Interested', 'Not_Interested']
attitude_classes = ['Positive', 'Negative', 'Not_Interacting']
jpl_action_classes = ['Handshake', 'Hug', 'Pet', 'Wave', 'Punch', 'Throw', 'Point', 'Gaze', 'Leave', 'No_Response']
harper_action_classes = ['Crash', 'Stop', 'Avoid', 'Touch', 'Kick', 'Punch']
coco_body_l_pair = [[0, 1], [0, 2], [1, 3], [2, 4],  # Head
                    [5, 7], [7, 9], [6, 8], [8, 10],  # Body
                    [5, 6], [11, 12], [5, 11], [6, 12],
                    [0, 5], [0, 6],
                    [11, 13], [12, 14], [13, 15], [14, 16]]
