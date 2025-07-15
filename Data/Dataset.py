import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from constants import head_point_num, hands_point_num, coco_body_point_num, device


class Pose_Dataset(Dataset):
    def __init__(self, data_path, sequence_length):
        super().__init__()
        self.files = os.listdir(os.path.join(data_path, 'pose_features'))
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.features, self.labels = [], []
        for file in self.files: self.get_graph_data_from_file(file)

    def get_graph_data_from_file(self, file):
        with open(os.path.join(self.data_path, 'pse_features', file), 'r') as f:
            feature_json = json.load(f)
        if not feature_json['frames']:
            return None
        first_id = feature_json['frames'][0]['frame_id'] if feature_json['frames'] else -1
        if first_id == -1:
            return
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        video_frame_num = len(feature_json['frames'])
        index, frame_num = 0, 0
        x_tensor = torch.empty((self.sequence_length, coco_body_point_num, 3))
        while frame_num < self.sequence_length:
            if index == video_frame_num:
                x_tensor[frame_num] = frame_feature
                frame_num += 1
            else:
                frame = feature_json['frames'][index]
                if frame['frame_id'] > first_id + frame_num:
                    x_tensor[frame_num] = frame_feature
                    frame_num += 1
                else:
                    index += 1
                    if frame['frame_id'] - first_id > self.sequence_length:
                        break
                    else:
                        frame_feature = torch.tensor(frame['keypoints'])[:coco_body_point_num]
                        frame_feature[frame['frame_id'] - first_id, :, 0] = torch.clamp(frame_feature[:, 0], min=0,
                                                                                        max=frame_width) / frame_width - 0.5
                        frame_feature[frame['frame_id'] - first_id, :, 1] = torch.clamp(frame_feature[:, 1], min=0,
                                                                                        max=frame_height) / frame_height - 0.5
                        frame_feature[frame['frame_id'] - first_id, :, 2] = frame_feature[:, 2]
                        x_tensor[frame_num] = frame_feature
                        frame_num += 1
        if frame_num == 0:
            return
        label = feature_json['intention_class'], feature_json['attitude_class'], feature_json['action_class']
        self.features.append(x_tensor)
        self.labels.append(label)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)


rgb_input_size = 224
heatmap_input_size = 56
sigma = 0.7


class SocialC3D_Dataset(Dataset):
    def __init__(self, data_path, dataset, modality, sequence_length):
        super().__init__()
        self.data_path = data_path
        self.json_files = os.listdir(os.path.join(data_path, 'pose_features'))
        self.dataset = dataset
        self.modality = modality
        self.sequence_length = sequence_length
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((rgb_input_size, rgb_input_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        data = []
        with open(os.path.join(self.data_path, 'pose_features', self.json_files[idx]), 'r') as f:
            feature_json = json.load(f)
        labels = feature_json['intention_class'], feature_json['attitude_class'], feature_json['action_class']
        frames = feature_json['frames']
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        first_id = frames[0]['frame_id']
        h, w = torch.meshgrid(
            torch.arange(int(frame_height), dtype=torch.float32),
            torch.arange(int(frame_width), dtype=torch.float32),
            indexing='ij'
        )

        if 'rgb' in self.modality:
            frame_index = 0
            video_file = feature_json['video_name']
            rgb_data = torch.zeros(3, self.sequence_length, rgb_input_size, rgb_input_size)
            if self.dataset == 'HARPER':
                for i in range(self.sequence_length):
                    if first_id + i < len(frames) and frames[first_id + frame_index]['frame_id'] == \
                            frames[first_id + i]['frame_id']:
                        image_id = '00000' + str(first_id + i)
                        image_id = image_id[-6:]
                        image = cv2.imread(
                            self.data_path + '/videos/' + video_file + '/' + video_file + '_' + image_id + '.jpg')
                        if image is None:
                            rgb_data[:, i, :, :] = rgb_data[:, i - 1, :, :]
                            continue
                        box_x, box_y, box_w, box_h = tuple(frames[first_id + frame_index]['box'])
                        box_x = box_x if box_x > 0 else 0
                        box_y = box_y if box_y > 0 else 0
                        box_w = box_w if box_w > 1 else 1
                        box_h = box_h if box_h > 1 else 1
                        image = image[int(box_y):int(box_y + box_h), int(box_x):int(box_x + box_w)]
                        image = cv2.resize(image, (rgb_input_size, rgb_input_size), interpolation=cv2.INTER_CUBIC)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = self.transform(image).unsqueeze(0)
                        rgb_data[:, i, :, :] = image
                        frame_index += 1
                    else:
                        rgb_data[:, i, :, :] = rgb_data[:, i - 1, :, :]
            else:
                cap = cv2.VideoCapture(self.data_path + '/videos/' + video_file)
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_id)
                for i in range(self.sequence_length):
                    if first_id + i < len(frames) and frames[first_id + frame_index]['frame_id'] == \
                            frames[first_id + i]['frame_id']:
                        s, image = cap.read()
                        if not s or image is None:
                            rgb_data[:, i, :, :] = rgb_data[:, i - 1, :, :]
                            continue
                        box_x, box_y, box_w, box_h = tuple(frames[first_id + frame_index]['box'])
                        box_x = box_x if box_x > 0 else 0
                        box_y = box_y if box_y > 0 else 0
                        box_w = box_w if box_w > 1 else 1
                        box_h = box_h if box_h > 1 else 1
                        image = image[int(box_y):int(box_y + box_h), int(box_x):int(box_x + box_w)]
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = self.transform(image).unsqueeze(0)
                        rgb_data[:, i, :, :] = image
                        frame_index += 1
                    else:
                        s, image = cap.read()
                        rgb_data[:, i, :, :] = rgb_data[:, i - 1, :, :]
                cap.release()
            data.append(rgb_data)
        else:
            data.append(0)

        if 'body' in self.modality:
            frame_index = 0
            body_data = torch.zeros(17, self.sequence_length, heatmap_input_size, heatmap_input_size)
            resize_shape = (heatmap_input_size, heatmap_input_size)
            for i in range(self.sequence_length):
                if first_id + i < len(frames) and frames[first_id + frame_index]['frame_id'] == frames[first_id + i][
                    'frame_id']:
                    for point_index in range(17):
                        point = frames[first_id + i]['keypoints'][point_index]
                        x, y, c = torch.tensor(point, dtype=torch.float32, device=device)
                        x = torch.clamp(x, 0, frame_width - 1)
                        y = torch.clamp(y, 0, frame_height - 1)
                        heatmap = c * torch.exp(-((h - y) ** 2 + (w - x) ** 2) / (2 * sigma ** 2))
                        heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1
                        heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=resize_shape,
                                                                  mode='bilinear', align_corners=False)
                        body_data[point_index, i, :, :] = heatmap.squeeze()
                    frame_index += 1
                else:
                    body_data[:, i, :, :] = body_data[:, i - 1, :, :]
            data.append(torch.from_numpy(body_data))
        else:
            data.append(0)

        if 'face' in self.modality:
            frame_index = 0
            face_data = torch.zeros(head_point_num, self.sequence_length, heatmap_input_size, heatmap_input_size)
            resize_shape = (heatmap_input_size, heatmap_input_size)
            for i in range(self.sequence_length):
                if first_id + i < len(frames) and frames[first_id + frame_index]['frame_id'] == frames[first_id + i][
                    'frame_id']:
                    for point_index in range(coco_body_point_num, coco_body_point_num + head_point_num):
                        point = frames[first_id + i]['keypoints'][point_index - coco_body_point_num]
                        x, y, c = torch.tensor(point, dtype=torch.float32, device=device)
                        x = torch.clamp(x, 0, frame_width - 1)
                        y = torch.clamp(y, 0, frame_height - 1)
                        heatmap = c * torch.exp(-((h - y) ** 2 + (w - x) ** 2) / (2 * sigma ** 2))
                        heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1
                        heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=resize_shape,
                                                                  mode='bilinear', align_corners=False)
                        face_data[point_index - coco_body_point_num, i, :, :] = heatmap.squeeze()
                    frame_index += 1
                else:
                    face_data[:, i, :, :] = face_data[:, i - 1, :, :]
            data.append(torch.from_numpy(face_data))
        else:
            data.append(0)

        if 'hand' in self.modality:
            frame_index = 0
            hand_data = torch.zeros(hands_point_num, self.sequence_length, heatmap_input_size, heatmap_input_size)
            resize_shape = (heatmap_input_size, heatmap_input_size)
            for i in range(self.sequence_length):
                if first_id + i < len(frames) and frames[first_id + frame_index]['frame_id'] == frames[first_id + i][
                    'frame_id']:
                    for point_index in range(coco_body_point_num + head_point_num,
                                             coco_body_point_num + head_point_num + hands_point_num):
                        point = frames[first_id + i]['keypoints'][point_index - coco_body_point_num - head_point_num]
                        x, y, c = torch.tensor(point, dtype=torch.float32, device=device)
                        x = torch.clamp(x, 0, frame_width - 1)
                        y = torch.clamp(y, 0, frame_height - 1)
                        heatmap = c * torch.exp(-((h - y) ** 2 + (w - x) ** 2) / (2 * sigma ** 2))
                        heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1
                        heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=resize_shape,
                                                                  mode='bilinear', align_corners=False)
                        hand_data[point_index - coco_body_point_num - head_point_num, i, :, :] = heatmap.squeeze()
                    frame_index += 1
                else:
                    hand_data[:, i, :, :] = hand_data[:, i - 1, :, :]
            data.append(torch.from_numpy(hand_data))
        else:
            data.append(0)

        if 'gaze' in self.modality:
            frame_index = 0
            gaze_data = torch.zeros(2, self.sequence_length, heatmap_input_size, heatmap_input_size)
            h = torch.tensor([i for i in range(int(frame_height))]).unsqueeze(1).expand(int(frame_height),
                                                                                        int(frame_width))
            w = torch.tensor([i for i in range(int(frame_width))]).unsqueeze(0).expand(int(frame_height),
                                                                                       int(frame_width))
            for i in range(self.sequence_length):
                if first_id + i < len(frames) and frames[first_id + frame_index]['frame_id'] == frames[first_id + i][
                    'frame_id']:
                    head_x, head_y, gaze_x, gaze_y = tuple(frames[first_id + i]['gaze'])
                    if head_x and head_y and gaze_x and gaze_y:
                        heatmap = torch.exp(-((h - head_y) ** 2 + (w - head_x) ** 2) / (2 * sigma ** 2))
                        heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1
                        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
                        heatmap = F.resize(heatmap, [heatmap_input_size, heatmap_input_size])
                        gaze_data[0, i, :, :] = heatmap.squeeze()
                        heatmap = torch.exp(-((h - gaze_y) ** 2 + (w - gaze_x) ** 2) / (2 * sigma ** 2))
                        heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1
                        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
                        heatmap = F.resize(heatmap, [heatmap_input_size, heatmap_input_size])
                        gaze_data[1, i, :, :] = heatmap.squeeze()
                    else:
                        head_x, head_y, c = tuple(frames[first_id + frame_index]['keypoints'][0])
                        head_x = max(0, min(head_x, frame_width))
                        head_y = max(0, min(head_y, frame_height))
                        heatmap = c * torch.exp(-((h - head_y) ** 2 + (w - head_x) ** 2) / (2 * sigma ** 2))
                        heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1
                        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
                        heatmap = F.resize(heatmap, [heatmap_input_size, heatmap_input_size])
                        gaze_data[0, i, :, :] = heatmap.squeeze()
                    frame_index += 1
                else:
                    gaze_data[:, i, :, :] = gaze_data[:, i - 1, :, :]
            data.append(torch.from_numpy(gaze_data))
        else:
            data.append(0)

        return data, labels

    def __len__(self):
        return len(self.json_files)
