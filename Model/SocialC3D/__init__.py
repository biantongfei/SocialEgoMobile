from .resnet3d_slowonly import ResNet3dSlowOnly
from .resnet3d_slowfast import ResNet3dPathway

from typing import Dict
from collections import OrderedDict
from mmengine.model.weight_init import constant_init, kaiming_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmengine.logging import MMLogger, print_log
from mmengine.runner.checkpoint import load_checkpoint
from torch import nn
import torch

from constants import head_point_num, hands_point_num, device, coco_body_l_pair, coco_body_point_num
from Model.Classifier import Chain_Classifier


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_uniform_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
    elif isinstance(m, nn.BatchNorm1d):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)


class RGBC3D(nn.Module):
    def __init__(self, dataset, pretrained=False):
        super().__init__()
        self.rgbc3d = ResNet3dSlowOnly(depth=50,
                                       conv1_kernel=(1, 7, 7),
                                       inflate=(0, 0, 1, 1))
        if pretrained:
            rgbc3d_weights = torch.load(pretrained)['state_dict']
            rgbc3d_weights = OrderedDict(
                [[k.split('backbone.')[-1],
                  v.cuda(device)] for k, v in rgbc3d_weights.items() if 'cls_head' not in k])
            self.rgbc3d.load_state_dict(rgbc3d_weights)
        else:
            self.rgbc3d.init_weights()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 16),
        )
        self.fc.apply(init_weights)
        self.classifier = Chain_Classifier(dataset, 16)
        self.classifier.apply(init_weights)

    def forward(self, x):
        x = x[0].to(device)
        x = self.rgbc3d(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x


class HeatmapC3D(nn.Module):
    def __init__(self, in_channels, dataset, pretrained=False):
        super().__init__()
        self.heatmapc3d = ResNet3dSlowOnly(in_channels=in_channels,
                                           base_channels=32,
                                           num_stages=3,
                                           out_indices=(2,),
                                           stage_blocks=(4, 6, 3),
                                           conv1_stride_s=1,
                                           pool1_stride_s=1,
                                           inflate=(0, 1, 1),
                                           spatial_strides=(2, 2, 2),
                                           temporal_strides=(1, 1, 1),
                                           dilations=(1, 1, 1))
        if pretrained:
            if in_channels == coco_body_point_num:
                heatmapc3d_weights = torch.load(pretrained)['state_dict']
                heatmap3d_weights = OrderedDict(
                    [[k.split('backbone.')[-1],
                      v.cuda(device)] for k, v in heatmapc3d_weights.items() if 'cls_head' not in k])
                self.heatmapc3d.load_state_dict(heatmap3d_weights)
        else:
            self.heatmapc3d.init_weights()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )
        self.classifier = Chain_Classifier(dataset, 16)

    def forward(self, x):
        x = x[0].to(device)
        x = self.heatmapc3d(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        y = self.classifier(x)
        return y


def padding(weight, new_shape):
    new_weight = weight.new_zeros(new_shape)
    new_weight[:, :weight.shape[1]] = weight
    return new_weight


class SocialC3D(nn.Module):
    def __init__(self,
                 pretrained,
                 dataset,
                 modality,
                 speed_ratio: int = 1,
                 channel_ratio: int = 4,
                 rgb_pathway: Dict = dict(
                     num_stages=4,
                     lateral=True,
                     lateral_infl=1,
                     lateral_activate=(0, 0, 1, 1),
                     fusion_kernel=7,
                     base_channels=64,
                     conv1_kernel=(1, 7, 7),
                     inflate=(0, 0, 1, 1),
                     with_pool2=False),
                 body_pathway: Dict = dict(
                     num_stages=3,
                     stage_blocks=(4, 6, 3),
                     lateral=True,
                     lateral_inv=True,
                     lateral_infl=16,
                     lateral_activate=(0, 1, 1),
                     fusion_kernel=7,
                     in_channels=17,
                     base_channels=32,
                     out_indices=(2,),
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_s=1,
                     conv1_stride_t=1,
                     pool1_stride_s=1,
                     pool1_stride_t=1,
                     inflate=(0, 1, 1),
                     spatial_strides=(2, 2, 2),
                     temporal_strides=(1, 1, 1),
                     dilations=(1, 1, 1),
                     with_pool2=False),
                 face_pathway: Dict = dict(
                     num_stages=3,
                     stage_blocks=(4, 6, 3),
                     lateral=True,
                     lateral_inv=True,
                     lateral_infl=16,
                     lateral_activate=(0, 1, 1),
                     fusion_kernel=7,
                     in_channels=head_point_num,
                     base_channels=32,
                     out_indices=(2,),
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_s=1,
                     conv1_stride_t=1,
                     pool1_stride_s=1,
                     pool1_stride_t=1,
                     inflate=(0, 1, 1),
                     spatial_strides=(2, 2, 2),
                     temporal_strides=(1, 1, 1),
                     dilations=(1, 1, 1),
                     with_pool2=False),
                 hand_pathway: Dict = dict(
                     num_stages=3,
                     stage_blocks=(4, 6, 3),
                     lateral=True,
                     lateral_inv=True,
                     lateral_infl=16,
                     lateral_activate=(0, 1, 1),
                     fusion_kernel=7,
                     in_channels=hands_point_num,
                     base_channels=32,
                     out_indices=(2,),
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_s=1,
                     conv1_stride_t=1,
                     pool1_stride_s=1,
                     pool1_stride_t=1,
                     inflate=(0, 1, 1),
                     spatial_strides=(2, 2, 2),
                     temporal_strides=(1, 1, 1),
                     dilations=(1, 1, 1),
                     with_pool2=False),
                 gaze_pathway: Dict = dict(
                     num_stages=3,
                     stage_blocks=(4, 6, 3),
                     lateral=True,
                     lateral_inv=True,
                     lateral_infl=16,
                     lateral_activate=(0, 1, 1),
                     fusion_kernel=7,
                     in_channels=2,
                     base_channels=32,
                     out_indices=(2,),
                     conv1_kernel=(1, 7, 7),
                     conv1_stride_s=1,
                     conv1_stride_t=1,
                     pool1_stride_s=1,
                     pool1_stride_t=1,
                     inflate=(0, 1, 1),
                     spatial_strides=(2, 2, 2),
                     temporal_strides=(1, 1, 1),
                     dilations=(1, 1, 1),
                     with_pool2=False)):
        super().__init__()
        self.modality = modality
        self.pretrained = pretrained
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if rgb_pathway['lateral']:
            rgb_pathway['speed_ratio'] = speed_ratio
            rgb_pathway['channel_ratio'] = channel_ratio
        if body_pathway['lateral']:
            body_pathway['speed_ratio'] = speed_ratio
            body_pathway['channel_ratio'] = channel_ratio
        if face_pathway['lateral']:
            face_pathway['speed_ratio'] = speed_ratio
            face_pathway['channel_ratio'] = channel_ratio
        if hand_pathway['lateral']:
            hand_pathway['speed_ratio'] = speed_ratio
            hand_pathway['channel_ratio'] = channel_ratio
        if gaze_pathway['lateral']:
            gaze_pathway['speed_ratio'] = speed_ratio
            gaze_pathway['channel_ratio'] = channel_ratio
        self.rgb_path = ResNet3dPathway(**rgb_pathway)
        self.body_path = ResNet3dPathway(**body_pathway)
        if pretrained:
            rgbc3d_weights = torch.load('pretrained_weights/rgb_only_20230228-576b9f86.pth')['state_dict']
            rgbc3d_weights = OrderedDict(
                [[k.split('backbone.')[-1],
                  v.cuda(device)] for k, v in rgbc3d_weights.items() if
                 'cls_head' not in k and k not in ['backbone.layer2.0.downsample.conv.weight',
                                                   'backbone.layer2.0.conv1.conv.weight',
                                                   'backbone.layer3.0.conv1.conv.weight',
                                                   'backbone.layer3.0.downsample.conv.weight']])
            self.rgbc3d.load_state_dict(rgbc3d_weights)

            bodyc3d_weights = torch.load('pretrained_weights/pose_only_20230228-fa40054e.pth')['state_dict']
            bodyc3d_weights = OrderedDict(
                [[k.split('backbone.')[-1],
                  v.cuda(device)] for k, v in bodyc3d_weights.items() if
                 'cls_head' not in k and k not in ['backbone.layer2.0.downsample.conv.weight',
                                                   'backbone.layer2.0.conv1.conv.weight',
                                                   'backbone.layer3.0.conv1.conv.weight',
                                                   'backbone.layer3.0.downsample.conv.weight']])
            self.body_path.load_state_dict(bodyc3d_weights)
        else:
            self.rgb_path.apply(init_weights)
            self.body_path.apply(init_weights)
        if 'face' in self.modality:
            self.face_path = ResNet3dPathway(**face_pathway)
        self.face_path.apply(init_weights)
        if 'hand' in self.modality:
            self.hand_path = ResNet3dPathway(**hand_pathway)
        self.hand_path.apply(init_weights)
        if 'gaze' in self.modality:
            self.gaze_path = ResNet3dPathway(**gaze_pathway)
        self.gaze_path.apply(init_weights)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048 + (len(self.modality) - 1) * 512, 1028),
            nn.BatchNorm1d(1028),
            nn.ReLU(),
            nn.Linear(1028, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.BatchNorm1d(16),
        )
        self.fc.apply(init_weights)
        self.classifier = Chain_Classifier(dataset, 16)
        self.classifier.apply(init_weights)

    def init_weights(self) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch separately.
            self.rgb_path.init_weights()
            self.body_path.init_weights()
            if 'face' in self.modality:
                self.face_path.init_weights()
            if 'hand' in self.modality:
                self.hand_path.init_weights()
            if 'gaze' in self.modality:
                self.gaze_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, representation) -> tuple:
        """Defines the computation performed at every call.

        Args:
            imgs (torch.Tensor): The input data.
            heatmap_imgs (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input
            samples extracted by the backbone.
        """
        # We assume base_channel for RGB and Pose are 64 and 32.
        imgs, body_heatmaps = x[0].to(device), x[1].to(device)
        if 'face' in self.modality:
            face_heatmaps = x[2].to(device)
        if 'hand' in self.modality:
            hand_heatmaps = x[3].to(device)
        if 'gaze' in self.modality:
            gaze_heatmaps = x[4].to(device)

        x_rgb = self.rgb_path.conv1(imgs)
        x_rgb = self.rgb_path.maxpool(x_rgb)

        # N x 64 x 8 x 56 x 56
        x_body = self.body_path.conv1(body_heatmaps)
        x_body = self.body_path.maxpool(x_body)

        if 'face' in self.modality:
            x_face = self.face_path.conv1(face_heatmaps)
            x_face = self.face_path.maxpool(x_face)
        if 'hand' in self.modality:
            x_hand = self.hand_path.conv1(hand_heatmaps)
            x_hand = self.hand_path.maxpool(x_hand)
        if 'gaze' in self.modality:
            x_gaze = self.gaze_path.conv1(gaze_heatmaps)
            x_gaze = self.gaze_path.maxpool(x_gaze)

        x_rgb = self.rgb_path.layer1(x_rgb)
        x_rgb = self.rgb_path.layer2(x_rgb)
        x_body = self.body_path.layer1(x_body)
        if 'face' in self.modality:
            x_face = self.face_path.layer1(x_face)
        if 'hand' in self.modality:
            x_hand = self.hand_path.layer1(x_hand)
        if 'gaze' in self.modality:
            x_gaze = self.gaze_path.layer1(x_gaze)

        if hasattr(self.rgb_path, 'layer2_lateral'):
            x_body_lateral = self.rgb_path.layer2_lateral(x_body)

        if hasattr(self.body_path, 'layer1_lateral'):
            x_body_rgb_lateral = self.body_path.layer1_lateral(x_rgb)

        if 'face' in self.modality and hasattr(self.face_path, 'layer1_lateral'):
            x_face_rgb_lateral = self.face_path.layer1_lateral(x_rgb)
            x_face_lateral = self.rgb_path.layer2_lateral(x_face)

        if 'hand' in self.modality and hasattr(self.hand_path, 'layer1_lateral'):
            x_hand_rgb_lateral = self.hand_path.layer1_lateral(x_rgb)
            x_hand_lateral = self.rgb_path.layer2_lateral(x_hand)

        if 'gaze' in self.modality and hasattr(self.gaze_path, 'layer1_lateral'):
            x_gaze_rgb_lateral = self.gaze_path.layer1_lateral(x_rgb)
            x_gaze_lateral = self.rgb_path.layer2_lateral(x_gaze)

        if hasattr(self.rgb_path, 'layer2_lateral'):
            x_body_lateral = x_body_lateral
            if 'face' in self.modality:
                x_body_lateral += x_face_lateral
            if 'hand' in self.modality:
                x_body_lateral += x_hand_lateral
            if 'gaze' in self.modality:
                x_body_lateral += x_gaze_lateral
            x_rgb = torch.cat((x_rgb, x_body_lateral), dim=1)

        if hasattr(self.body_path, 'layer1_lateral'):
            x_body = torch.cat((x_body, x_body_rgb_lateral), dim=1)

        if 'face' in self.modality and hasattr(self.face_path, 'layer1_lateral'):
            x_face = torch.cat((x_face, x_face_rgb_lateral), dim=1)

        if 'hand' in self.modality and hasattr(self.hand_path, 'layer1_lateral'):
            x_hand = torch.cat((x_hand, x_hand_rgb_lateral), dim=1)

        if 'gaze' in self.modality and hasattr(self.gaze_path, 'layer1_lateral'):
            x_gaze = torch.cat((x_gaze, x_gaze_rgb_lateral), dim=1)

        x_rgb = self.rgb_path.layer3(x_rgb)
        x_body = self.body_path.layer2(x_body)
        if 'face' in self.modality:
            x_face = self.face_path.layer2(x_face)
        if 'hand' in self.modality:
            x_hand = self.hand_path.layer2(x_hand)
        if 'gaze' in self.modality:
            x_gaze = self.gaze_path.layer2(x_gaze)

        if hasattr(self.rgb_path, 'layer3_lateral'):
            x_body_lateral = self.rgb_path.layer3_lateral(x_body)

        if hasattr(self.body_path, 'layer2_lateral'):
            x_body_rgb_lateral = self.body_path.layer2_lateral(x_rgb)

        if 'face' in self.modality and hasattr(self.face_path, 'layer1_lateral'):
            x_face_rgb_lateral = self.face_path.layer2_lateral(x_rgb)
            x_face_lateral = self.rgb_path.layer3_lateral(x_face)

        if 'hand' in self.modality and hasattr(self.hand_path, 'layer1_lateral'):
            x_hand_rgb_lateral = self.hand_path.layer2_lateral(x_rgb)
            x_hand_lateral = self.rgb_path.layer3_lateral(x_hand)

        if 'gaze' in self.modality and hasattr(self.gaze_path, 'layer1_lateral'):
            x_gaze_rgb_lateral = self.gaze_path.layer2_lateral(x_rgb)
            x_gaze_lateral = self.rgb_path.layer3_lateral(x_gaze)

        if hasattr(self.rgb_path, 'layer3_lateral'):
            x_body_lateral = x_body_lateral
            if 'face' in self.modality:
                x_body_lateral += x_face_lateral
            if 'hand' in self.modality:
                x_body_lateral += x_hand_lateral
            if 'gaze' in self.modality:
                x_body_lateral += x_gaze_lateral
            x_rgb = torch.cat((x_rgb, x_body_lateral), dim=1)

        if hasattr(self.body_path, 'layer2_lateral'):
            x_body = torch.cat((x_body, x_body_rgb_lateral), dim=1)

        if 'face' in self.modality and hasattr(self.face_path, 'layer2_lateral'):
            x_face = torch.cat((x_face, x_face_rgb_lateral), dim=1)

        if 'hand' in self.modality and hasattr(self.hand_path, 'layer2_lateral'):
            x_hand = torch.cat((x_hand, x_hand_rgb_lateral), dim=1)

        if 'gaze' in self.modality and hasattr(self.gaze_path, 'layer2_lateral'):
            x_gaze = torch.cat((x_gaze, x_gaze_rgb_lateral), dim=1)

        x_rgb = self.rgb_path.layer4(x_rgb)
        x_body = self.body_path.layer3(x_body)
        if 'face' in self.modality:
            x_face = self.face_path.layer3(x_face)
        if 'hand' in self.modality:
            x_hand = self.hand_path.layer3(x_hand)
        if 'gaze' in self.modality:
            x_gaze = self.gaze_path.layer3(x_gaze)

        x_rgb = self.avg_pool(x_rgb)
        x_rgb = x_rgb.view(x_rgb.shape[0], -1)
        x_body = self.avg_pool(x_body)
        x_body = x_body.view(x_body.shape[0], -1)
        x = [x_rgb, x_body]
        if 'face' in self.modality:
            x_face = self.avg_pool(x_face)
            x_face = x_face.view(x_face.shape[0], -1)
            x.append(x_face)
        if 'hand' in self.modality:
            x_hand = self.avg_pool(x_hand)
            x_hand = x_hand.view(x_hand.shape[0], -1)
            x.append(x_hand)
        if 'gaze' in self.modality:
            x_gaze = self.avg_pool(x_gaze)
            x_gaze = x_gaze.view(x_gaze.shape[0], -1)
            x.append(x_gaze)
        x = torch.cat(x, dim=1)
        x = self.fc(x)
        return x if representation else self.classifier(x)
