from Model.SocialEgoMobile import SocialEgoMobile
from Model.SocialC3D import SocialC3D
from Model.InfoNCE import InfoNCE
from constants import device
from Data.Dataset import Pose_Dataset, filter_not_interacting_sample, SocialC3D_Dataset
from PoseDataLoader import Pose_DataLoader

import torch
from torch.nn import functional
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Train SocialEgoMobile')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--distillation',  action="store_true")
    parser.add_argument('--save_weights', type=str, default=None)
    return parser.parse_args()


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get_student_dataloaders(config):
    data_path = config['data']['path']
    sequence_length = config['data']['sequence_length']
    trainset = Pose_Dataset(data_path + 'train/', sequence_length)
    valset = Pose_Dataset(data_path + 'validation/', sequence_length)
    testset = Pose_Dataset(data_path + 'test/', sequence_length)

    batch_size = config['train']['batch_size']
    train_loader = Pose_DataLoader(trainset, batch_size=batch_size, sequence_length=sequence_length)
    val_loader = Pose_DataLoader(valset, batch_size=batch_size, sequence_length=sequence_length)
    test_loader = Pose_DataLoader(testset, batch_size=batch_size, sequence_length=sequence_length)

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


def create_socialegomobile(args, config):
    model = SocialEgoMobile(
        sequence_length=config['data']['sequence_length'],
        dataset=args.dataset,
        keypoint_hidden_dim=config['model']['keypoint_hidden_dim'],
        gcn_layers=config['model']['gcn_layers'],
        lstm_hidden_dim=config['model']['lstm_hidden_dim'],
        lstm_layers=config['model']['lstm_layers'],
        fc_hidden1=config['model']['fc_hidden1'],
        fc_hidden2=config['model']['fc_hidden2'],
    ).to(device)
    return model


def create_socialc3d(args, config):
    model = SocialC3D(
        dataset=args.dataset,
        modality=config['data']['modality'],
    ).to(device)
    return model


@torch.no_grad()
def evaluate_model(model, dataloader, task):
    model.eval()

    metrics = {"int": {"true": [], "pred": []}, "att": {"true": [], "pred": []}, "act": {"true": [], "pred": []}}

    for inputs, (int_labels, att_labels, act_labels) in tqdm(dataloader, dynamic_ncols=True, desc=task + " Evaluating"):
        int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
        int_outputs, att_outputs, act_outputs, _ = model(inputs)

        metrics["int"]["true"].extend(int_labels.tolist())
        metrics["int"]["pred"].extend(torch.argmax(torch.softmax(int_outputs, dim=1), dim=1).tolist())

        att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)

        metrics["att"]["true"].extend(att_labels.tolist())
        metrics["att"]["pred"].extend(torch.argmax(torch.softmax(att_outputs, dim=1), dim=1).tolist())

        metrics["act"]["true"].extend(act_labels.tolist())
        metrics["act"]["pred"].extend(torch.argmax(torch.softmax(act_outputs, dim=1), dim=1).tolist())

    results = {}
    for key in metrics:
        y_true, y_pred = torch.tensor(metrics[key]["true"]), torch.tensor(metrics[key]["pred"])
        acc = (y_pred == y_true).sum().item() / len(y_true)
        f1 = f1_score(y_true, y_pred, average='weighted')
        results[key] = {"acc": acc * 100, "f1": f1 * 100}

    print(f"{task} Results -> int_acc: {results['int']['acc']:.2f}, int_f1: {results['int']['f1']:.2f}, "
          f"att_acc: {results['att']['acc']:.2f}, att_f1: {results['att']['f1']:.2f}, "
          f"act_acc: {results['act']['acc']:.2f}, act_f1: {results['act']['f1']:.2f}")

    return results


def independent_train(args, config):
    train_loader, val_loader, test_loader = get_student_dataloaders(config)
    socialegomobile = create_socialegomobile(args, config)
    if config['train']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(socialegomobile.parameters(), lr=config['train']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    socialegomobile = socialegomobile

    for epoch in range(config['train']['independent_training_epochs']):
        train_loader = tqdm(train_loader, dynamic_ncols=True)
        for inputs, (int_labels, att_labels, act_labels) in train_loader:
            int_labels, att_labels, act_labels = int_labels.to(dtype=torch.long, device=device), att_labels.to(
                dtype=torch.long, device=device), act_labels.to(dtype=torch.long, device=device)
            (int_outputs, att_outputs, act_outputs), _ = socialegomobile(inputs)
            loss_1 = functional.cross_entropy(int_outputs, int_labels)
            loss_2 = functional.cross_entropy(att_outputs, att_labels)
            loss_3 = functional.cross_entropy(act_outputs, act_labels)
            total_loss = loss_1 + loss_2 + loss_3
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        evaluate_model(socialegomobile, val_loader, 'validation')

    print("Testing model on test set...")
    evaluate_model(socialegomobile, val_loader, 'Test')

    if args.save_weights:
        torch.save(socialegomobile.state_dict(), f"{args.save_weights}/socialegonet.pt")
        print(f"Model saved at {args.save_weights}/socialegonet.pt")


def knowledge_distillation(args, config):
    student_train_loader, student_val_loader, student_test_loader = get_student_dataloaders(config)
    teacher_train_loader, teacher_val_loader, teacher_test_loader = get_teacher_dataloaders(args, config)
    socialegomobile = create_socialegomobile(args, config)
    socialc3d = create_socialc3d(args, config)
    info_nce = InfoNCE(temperature=config['train']['infonce_temperature'])
    if config['train']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(socialegomobile.parameters(), lr=config['train']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['train']['step_size'], gamma=config['train']['gamma'])

    for epoch in range(config['train']['num_epochs']):
        if epoch < config['train']['warmup_epochs']:
            student_train_loader.keypoint_drop_rate = config['data']['keypoint_drop_rate'] * (
                    epoch / config['train']['warmup_epochs'])
            student_train_loader.frame_drop_rate = config['data']['frame_drop_rate'] * (
                    epoch / config['train']['warmup_epochs'])
        else:
            student_train_loader.keypoint_drop_rate = config['data']['keypoint_drop_rate']
            student_train_loader.frame_drop_rate = config['data']['frame_drop_rate']
        student_train_loader = tqdm(student_train_loader, dynamic_ncols=True)
        for (student_inputs, (int_labels, att_labels, act_labels)), (teacher_inputs, _) in zip(teacher_train_loader,
                                                                                               student_train_loader):
            int_labels, att_labels, act_labels = int_labels.to(dtype=torch.long, device=device), att_labels.to(
                dtype=torch.long, device=device), act_labels.to(dtype=torch.long, device=device)
            (int_outputs, att_outputs, act_outputs), student_social_feature = socialegomobile(student_inputs,
                                                                                              representation=True)
            loss_1 = functional.cross_entropy(int_outputs, int_labels)
            loss_2 = functional.cross_entropy(att_outputs, att_labels)
            loss_3 = functional.cross_entropy(act_outputs, act_labels)
            with torch.no_grad():
                teacher_social_feature = socialc3d(teacher_inputs, only_representation=True)
            loss_mi = info_nce(student_social_feature, teacher_social_feature.detach())
            total_loss = loss_1 + loss_2 + loss_3 + args.mi_loss_weight * loss_mi
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        evaluate_model(socialegomobile, student_val_loader, 'validation')

    print("Testing model on test set...")
    evaluate_model(socialegomobile, student_test_loader, 'Test')

    if args.save_weights:
        torch.save(socialegomobile.state_dict(), f"{args.save_weights}/socialegonet.pt")
        print(f"Model saved at {args.save_weights}/socialegonet.pt")


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.cfg)
    if args.distillation:
        knowledge_distillation(args, config)
    else:
        independent_train(args, config)
