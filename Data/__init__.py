from Dataset import Pose_Dataset, SocialC3D_Dataset
from PoseDataLoader import Pose_DataLoader
from torch.utils.data import DataLoader


def get_student_dataloaders(config):
    data_path = config['data']['path']
    sequence_length = config['data']['sequence_length']
    trainset = Pose_Dataset(data_path + 'train/', sequence_length)
    valset = Pose_Dataset(data_path + 'validation/', sequence_length)
    testset = Pose_Dataset(data_path + 'test/', sequence_length)

    batch_size = config['train']['batch_size']
    train_loader = Pose_DataLoader(trainset, batch_size=batch_size, sequence_length=sequence_length,
                                   keypoint_drop_rate=config['data']['keypoint_drop_rate'],
                                   frame_drop_rate=config['data']['frame_drop_rate'])
    val_loader = Pose_DataLoader(valset, batch_size=batch_size, sequence_length=sequence_length,
                                 keypoint_drop_rate=config['data']['keypoint_drop_rate'],
                                 frame_drop_rate=config['data']['frame_drop_rate'])
    test_loader = Pose_DataLoader(testset, batch_size=batch_size, sequence_length=sequence_length,
                                  keypoint_drop_rate=config['data']['keypoint_drop_rate'],
                                  frame_drop_rate=config['data']['frame_drop_rate'])

    return train_loader, val_loader, test_loader


def get_teacher_dataloaders(args, config):
    data_path = config['data']['path']
    sequence_length = config['data']['sequence_length']
    trainset = SocialC3D_Dataset(data_path=data_path + 'train/', dataset=args.dataset,
                                 modality=config['data']['modality'], sequence_length=sequence_length)
    valset = SocialC3D_Dataset(data_path=data_path + 'validation/', dataset=args.dataset,
                               modality=config['data']['modality'], sequence_length=sequence_length)
    testset = SocialC3D_Dataset(data_path=data_path + 'test/', dataset=args.dataset,
                                modality=config['data']['modality'], sequence_length=sequence_length)

    batch_size = config['train']['batch_size']
    train_loader = DataLoader(trainset, batch_size=batch_size)
    val_loader = DataLoader(valset, batch_size=batch_size)
    test_loader = DataLoader(testset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def filter_not_interacting_sample(att_y_true, att_y_output):
    mask = (att_y_true != 2)
    return att_y_true[mask], att_y_output[mask]
