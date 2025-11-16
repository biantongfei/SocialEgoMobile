from Model.InfoNCE import InfoNCE
from Model import create_socialegomobile, create_socialc3d
from constants import device
from Data import get_student_dataloaders, get_teacher_dataloaders
from tools import load_config, evaluate_model

import torch
from torch.nn import functional
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train SocialEgoMobile')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--distillation', action="store_true")
    parser.add_argument('--save_weights', type=str, default=None)
    return parser.parse_args()


def independent_train(args, config):
    train_loader, val_loader, test_loader = get_student_dataloaders(config)
    socialegomobile = create_socialegomobile(args, config)
    if config['train']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(socialegomobile.parameters(), lr=config['train']['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['train']['independent_training_epochs'])

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
    scheduler = CosineAnnealingLR(optimizer, T_max=config['train']['independent_training_epochs'])

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
