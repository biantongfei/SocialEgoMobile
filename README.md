# SocialEgoMobile

This is the official repo for the ACM MM 25 paper:

[Robust Understanding of Human-Robot Social Interactions through Multimodal Distillation](https://arxiv.org/abs/2505.06278) [![arXiv](https://img.shields.io/badge/arXiv-2505.06278-b31b1b.svg)](https://arxiv.org/abs/2505.06278)

<div align="center">
    <img src="docs/teaser_bg.png", height="280" alt>
</div>

Abstract:
The need for social robots and agents to interact with and assist users is growing steadily. To naturally interact with
humans, they need to understand and analyse socially interactive scenes from their (robot's) perspective. Works that
model social situations between humans and agents are few; and even those existing ones are often too computationally
intensive to be suitable for deployment in real-time or on real-world scenarios with limited available information. We
propose a robust knowledge distillation framework that models social interactions through various multimodal cues, yet
is robust against incomplete and noisy information during inference. Our teacher model is trained with multimodal
input (body, face and hand gestures, gaze, raw images) that transfers knowledge to a student model that relies solely on
body pose. Extensive experiments on two publicly available human-robot interaction datasets demonstrate that our student
model achieves an average accuracy gain of 14.75% over relevant baselines on multiple downstream social understanding
tasks even with up to 51% of its input being corrupted. The student model is highly efficient: it is <1% in size of
the teacher model in terms of parameters and its latency is 11.9% of the teacher model.

<div align="center">
    <img src="docs/distillation_bg.png", height="650" alt>
</div>

Our knowledge distillation framework uses SocialC3D as the teacher model, which fuses raw images, body, face, hand
gestures, and gaze. Each modality is processed by a ResNet [1] and integrated via lateral connections and late fusion,
producing a high-quality social representation for downstream tasks. The lightweight student model, SocialEgoMobile,
uses only corrupted body pose. It consists of a two-layer GAT [2] and a single-layer Bi-LSTM [3] to extract social
representations. The framework distillates knowledge from the teacher model by maximising the mutual information [4] of
the
social representations output by the teacher and student model. Whole body pose features were extracted
using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) and gaze features were extracted
using [MCGaze](https://github.com/zgchen33/MCGaze).

## Result

###Performance

|                                   | Params (M) | Latency (ms)  | Intent Acc. | Attitude Acc. | Action Acc. |
|-----------------------------------|------------|---------------|-------------|---------------|-------------|
| ST-GCN<sup>+</sup> [5]            | 42.86      | Δ1 + 10.31    | 86.90       | 76.19         | 71.43       |
| ST-TR<sup>+</sup> [6]             | 58.48      | Δ1 + 14.33    | 79.76       | 59.52         | 48.81       |
| MS-G3D<sup>+</sup> [7]            | 48.82      | Δ1 + 13.74    | 88.10       | 80.95         | 76.19       |
| SocialEgoNet<sup>+</sup> [8]      | 37.78      | Δ1 + 10.05    | 86.90       | 77.38         | 71.43       |
| **SocialEgoC3D (our teacher)**    | 48.49      | Δ1 + 22.34    | **92.85**   | **88.10**     | **82.14**   |
| **SocialEgoMobile (our student)** | **0.43**   | **Δ2 + 0.19** | 82.14       | 71.43         | 67.86       |

Table.1 Performance on [JPL-Social](https://github.com/biantongfei/SocialEgoNet)

|                                   | Intent Acc. | Attitude Acc. | Action Acc. |
|-----------------------------------|-------------|---------------|-------------|
| ST-GCN<sup>+</sup> [5]            | 86.54       | 73.08         | 78.85       |
| ST-TR<sup>+</sup> [6]             | 75.00       | 65.38         | 59.61       |
| MS-G3D<sup>+</sup> [7]            | 90.38       | 78.85         | 80.77       |
| SocialEgoNet<sup>+</sup> [8]      | 86.54       | 75.00         | 78.85       |
| **SocialEgoC3D (our teacher)**    | **96.15**   | **82.69**     | **88.46**   |
| **SocialEgoMobile (our student)** | 69.23       | 44.23         | 52.19       |

Table.2 Performance on [HARPER](https://github.com/intelligolabs/HARPER)

Comparison of SocialC3D and SocialEgoMobile with state-of-the-art methods on
the [JPL-Social](https://github.com/biantongfei/SocialEgoNet) and [HARPER](https://github.com/intelligolabs/HARPER)
dataset. SocialEgoMobile relies solely on clean body pose features as input. '+' indicates that the model uses raw image
and gaze information. SocialEgoMobile only use body pose as input. Δ1 denotes the time to extract whole-body pose and
gaze features from a single frame, which is 4.96 ms under our experimental setup. Δ2 refers to the extraction time for
body pose features, which is 3.06 ms. As a teacher model, SocialC3D showed superior performance on both datasets and all
subtasks. As a student model, SocialEgoMobile showed a notable decrease in performance, especially on the more difficult
action prediction task and on the HARPER dataset. However, it shows an advantage in computational efficiency with a
model size of ~1% of SocialC3D and its latency of 11.9% of SocialC3D.

### Robustness Analysis

To address the practical challenges faced by mobile socially intelligent agents from an egocentric perspective, such as
occlusion of user body parts and pose estimation error, we deliberately introduced random spatiotemporal corruption into
the input of the student model SocialEgoMobile. The goal is to improve robustness by using the multimodal and
uncorrupted knowledge extracted from the teacher model, SocialC3D, to supervise the learning of the student model with
corrupted inputs.

<div align="center">
    <img src="docs/corrupted_bar_jpl_bg.png", height="250" alt>
</div>

<div align="center">
    <img src="docs/corrupted_bar_harper_bg.png", height="250" alt>
</div>

Knowledge distillation (KD) consistently improves the performance of the student model, SocialEgoMobile, under
Individual and simultaneous spatio-temporal corruption on all three downstream tasks, interaction intent, attitude, and
social action forecast. Improvements on downstream task accuracy through distillation are labelled.

## Data

The datasets used in this paper can be downloaded here:
JPL_Social ([pose](https://drive.google.com/file/d/1gpH_T60e99cR_x4C5B2YKvPPa99rBzic/view?usp=sharing), [videos](http://michaelryoo.com/jpl-interaction.html))
and
HARPER ([pose](https://drive.google.com/file/d/1lczAS_XYBwN4jWYMIgzRfXFGFaeA98Xf/view?usp=sharing), [images](https://github.com/intelligolabs/HARPER)).
A detailed description of the datasets can be found here: [JPL-Social](https://github.com/biantongfei/SocialEgoNet)
and [HARPER](https://github.com/intelligolabs/HARPER).

## Train and Test

The pretrained weights of PoseC3D on Kinect-400 can be
downloaded [here](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/rgbpose_conv3d/README.md)
and the pretrained weights of SocialC3D and SocialEgoMobile can be
downloaded [here](https://drive.google.com/drive/folders/1j2_fad-rvbNG-Sy9VUzJ_VlRDqQ-r35B?usp=sharing).

To train and test SocialC3D and SocialEgoMobile, you need download the data and save it under the current project path.

To train a new SocialC3D, run

```
python scripts/train_SocialC3D.py --cfg config/SocialC3D.yaml --dataset JPL --pretrained
```

To train a new SocialEgoMobile independently, run

```
python scripts/train_SocialEgoMobile.py --cfg config/SocialEgoMobile.yaml --dataset JPL 
```

To train a new SocialEgoMobile via knowledge distillation, run

```
python scripts/train_SocialEgoMobile.py --cfg config/train_SocialEgoMobile.yaml --dataset JPL --distillation
```

To test the pretrained weights of SocialC3D, run

```
python scripts/test.py --cfg config/SocialEgoMobile.yaml --check_point weights/jpl_socialc3d_rgb_body_face_hand_gaze.pt.pt
```

To test the pretrained weights of SocialEgoMobile, run

```
python scripts/test.py --cfg config/SocialC3D.yaml --check_point weights/socialegomobile_jpl.pt
```

## Citation

Please cite the following paper if you use this repository in your research.

```
@INPROCEEDINGS{bian2025robust,
  author={Bian, Tongfei and Chollet, Mathieu and Guha, Tanaya},
  booktitle={Proceedings of the 33nd ACM International Conference on Multimedia}, 
  title={Robust Understanding of Human-Robot Social Interactions through Multimodal Distillation}, 
  year={2025}
}
```

## Refrences

```
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition. 770–778.
[2] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph Attention Networks. In International Conference on Learning Representations.
[3] Alex Graves and Jürgen Schmidhuber. 2005. Framewise phoneme classification with bidirectional LSTM networks. In Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005., Vol. 4. IEEE, 2047–2052.
[4] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018).
[5] Sijie Yan, Yuanjun Xiong, and Dahua Lin. 2018. Spatial temporal graph convolutional networks for skeleton-based action recognition. In Proceedings of the AAAI conference on artificial intelligence, Vol. 32.
[6] Chiara Plizzari, Marco Cannici, and Matteo Matteucci. 2021. Spatial temporal transformer network for skeleton-based action recognition. In Pattern recognition. ICPR international workshops and challenges: virtual event, January 10–15, 2021, Proceedings, Part III. Springer, 694–701.
[7] Ziyu Liu, Hongwen Zhang, Zhenghao Chen, Zhiyong Wang, and Wanli Ouyang. 2020. Disentangling and unifying graph convolutions for skeleton-based action recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 143–152.
[8] Tongfei Bian, Yiming Ma, Mathieu Chollet, Victor Sanchez, and Tanaya Guha. 2025. Interact with me: Joint Egocentric Forecasting of Intent to Interact, Attitude and Social Actions. In Proceedings of the IEEE International Conference on Multimedia and Expo (ICME).
```
