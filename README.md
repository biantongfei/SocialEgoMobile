# SocialEgoMobile

This is the official repo for the paper:

[Robust Understanding of Human-Robot Social Interactions through Multimodal Distillation](https://arxiv.org/abs/2412.16698) [![arXiv](https://img.shields.io/badge/arXiv-2412.16698-b31b1b.svg)](https://arxiv.org/abs/2412.16698)

<div align="center">
    <img src="docs/teaser_bg.png", height="280" alt>
</div>

Abstract:
The need for social robots and agents to interact and assist humans is growing steadily. To be able to successfully
interact with humans, they need to understand and analyse socially interactive scenes from their (robot's) perspective.
Works that model social situations between humans and agents are few; and even those existing ones are often too
computationally intensive to be suitable for deployment in real time or on real world scenarios with limited available
information. We propose a robust knowledge distillation framework that models social interactions through various
multimodal cues, yet is robust against incomplete and noisy information during inference. Our teacher model is trained
with multimodal input (body, face and hand gestures, gaze, raw images) that transfers knowledge to a student model that
relies solely on body pose. Extensive experiments on two publicly available human-robot interaction datasets demonstrate
that the our student model achieves an average accuracy gain of 14.75% over relevant baselines on multiple downstream
social understanding task even with up to 51\% of its input being corrupted. The student model is highly efficient: it
is <1% in size of the teacher model in terms of parameters and uses ~ 0.55‰ FLOPs of that in the teacher model.

<div align="center">
    <img src="docs/distillation_bg.png", height="650" alt>
</div>

Our knowledge distillation framework uses SocialC3D as the teacher model, which fuses raw images, body, face, hand
gestures, and gaze. Each modality is processed by a ResNet [1] and integrated via lateral connections and late fusion,
producing a high-quality social representation for downstream tasks. The lightweight student model, SocialEgoMobile,
uses only corrupted body pose. It consists of a two-layer GAT [2] and a single-layer Bi-LSTM [3] to extract social
representations. The framework distillates knowledge from the teacher model by maximising the mutual information of the
social representations output by the teacher and student model.

## Result

**Performance**

| | | | Intent Acc. | Attitude Acc. | Action Acc. |
| | Params (M) | FLOPs (M) | Intent Acc. | Attitude Acc. | Action Acc. |
|------------------------------|------------|-----------|-------------|---------------|-------------|
| ST-GCN<sup>+</sup> [4]       | 9.42 | Δ + 1.40 | 87.30 | 87.84 | 65.19 |
| ST-TR<sup>+</sup> [5]        | 8.78 | Δ + 3.47 | 83.92 | 84.34 | 67.18 |
| MS-G3D<sup>+</sup> [6]       | 12.82 | Δ + 4.74 | 90.02 | 90.11 | 73.29 |
| SocialEgoNet<sup>+</sup> [7] | 12.82 | Δ + 4.74 | 90.02 | 90.11 | 73.29 |
| **SocialEgoC3D (ours)**      | 3.18 | Δ + 0.56 | 88.43 | 88.99 | 69.57 |
| **SocialEgoMobile (ours)**   | 9.42 | Δ + 1.40 | 87.30 | 87.84 | 65.19 |

Δ refers to the time to extract the whole-body pose keypoints. In our
case, [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) takes 1.27 ms to extract whole-body pose features from an
annotated frame.

## Data

This dataset used in the paper, JPL-Social, is based on
the [JPL First-Person Interaction dataset (JPL-Interaction dataset)](http://michaelryoo.com/jpl-interaction.html).
JPL-Social has the whole-body key points of people in videos getting
from [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), including body, face and hand, and the additional social
intention attitude and action classes of each person in videos. JPL-Social can be downloaded
from [here](https://drive.google.com/file/d/1gpH_T60e99cR_x4C5B2YKvPPa99rBzic/view?usp=drive_link).

Metadata

```
{
  "video_name": Name of the video this person is in (string),
  "frame_size": [width of the frame (integer), height of the frame (integer)] (array),
  "video_frames_number": Number of video frames,
  "detected_frames_number": Number of frames the person is detected by Alphapose,
  "person_id": Id of this person in the video (integer),
  "attitude_class": attitude id of this person (integer),
  "action_class": action id of this person (integer),
  "frames": [{
      "frame_id": Id of video frame (Integer),
      "key points": [[x (float), y (float), score (float)], ..., [x, y, score]] (array),
      "score": Confidence score of this person this frame (float),
      "box": [x (float), y (float), width (float), height (float)] - bounding box information,
      },
      ...,
      ]
}
```

## Train and Test

To train and test SocialEgoNet on JPL-Social, you need download the data and save it under the current project path.

To train a new SocialEgoNet, run

```
python scripts/train.py --cfg config/train.yaml
```

To test the pretrained weights on JPL-Social, run

```
python scripts/test.py --cfg config/test.yaml --check_point weights/socialegonet_jpl.pt
```

## Citation

Please cite the following paper if you use this repository in your reseach.

```
@INPROCEEDINGS{bian2024interact,
  author={Bian, Tongfei and Ma, Yiming and Chollet, Mathieu and Sanchez, Victor and Guha, Tanaya},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={Interact with me: Joint Egocentric Forecasting of Intent to Interact, Attitude and Social Actions}, 
  year={2024}
}
```

## Refrences

```
[1] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri, “A closer look at spatiotemporal convolutions for action recognition,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2018, pp. 6450–6459. 2, 4, 5
[2] Sijie Yan, Yuanjun Xiong, and Dahua Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” in Proceedings of the AAAI conference on artificial intelligence, 2018, vol. 32. 2, 4, 5
[3] Haodong Duan, Jiaqi Wang, Kai Chen, and Dahua Lin, “Dg-stgcn: Dynamic spatial-temporal modeling for skeleton-based action recognition,”arXiv preprint arXiv:2210.05895, 2022. 4, 5
[4] Ziyu Liu, Hongwen Zhang, Zhenghao Chen, Zhiyong Wang, and Wanli Ouyang, “Disentangling and unifying graph convolutions for skeleton-based action recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. 4, 5
```
