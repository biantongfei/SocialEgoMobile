from constants import device
from Data import get_teacher_dataloaders
from Model import create_socialc3d
from tools import evaluate_model
import argparse
import tqdm
import torch
from torch.nn import functional


def parse_args():
    parser = argparse.ArgumentParser(description='Train SocialEgoMobile')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--save_weights', type=str, default=None)
    return parser.parse_args()


def load_param_groups(net, config):
    param_groups = config['train']['param_groups']
    for name, param in net.named_parameters():
        if "path" in name:
            if name in ['rgb_path.layer3.0.conv1.conv.weight', 'rgb_path.layer4.0.conv1.conv.weight',
                        'rgb_path.layer3.0.downsample.conv.weight', 'rgb_path.layer4.0.downsample.conv.weight',
                        'rgb_path.layer2_lateral.conv.weight', 'rgb_path.layer3_lateral.conv.weight']:
                param_groups[1]["params"].append(param)
            elif name.endswith('_path.layer2.0.downsample.conv.weight') or name.endswith(
                    '_path.layer3.0.downsample.conv.weight') or name.endswith(
                '_path.layer2.0.conv1.conv.weight') or name.endswith(
                '_path.layer3.0.conv1.conv.weight') or name.endswith(
                "_path.layer1_lateral.conv.weight") or name.endswith("_path.layer1_lateral.bn.weight") or name.endswith(
                "_path.layer1_lateral.bn.bias") or name.endswith(
                "_path.layer1_lateral.bn.running_mean") or name.endswith(
                "_path.layer1_lateral.bn.running_var") or name.endswith(
                "_path.layer2_lateral.conv.weight") or name.endswith("_path.layer2_lateral.bn.weight") or name.endswith(
                "_path.layer2_lateral.bn.bias") or name.endswith(
                '_path.layer2_lateral.bn.running_mean') or name.endswith('_path.layer2_lateral.bn.running_var'):
                param_groups[0]["params"].append(param)
            else:
                param_groups[2]["params"].append(param)
        else:
            param_groups[0]["params"].append(param)


def train(args, config):
    train_loader, val_loader, test_loader = get_teacher_dataloaders(args, config)
    socialc3d = create_socialc3d(args, config)
    if config['train']['optimizer'] == 'sgd':
        if args.pretrained:
            param_groups = load_param_groups(socialc3d, config)
            optimizer = torch.optim.SGD(param_groups, weight_decay=config['train']['weight_decay'])
        else:
            optimizer = torch.optim.SGD(socialc3d.parameters(), weight_decay=config['train']['weight_decay'],
                                        lr=config['train']['learning_rate'])

    for epoch in range(config['train']['epochs']):
        train_loader = tqdm(train_loader, dynamic_ncols=True)
        for inputs, (int_labels, att_labels, act_labels) in train_loader:
            int_labels, att_labels, act_labels = int_labels.to(dtype=torch.long, device=device), att_labels.to(
                dtype=torch.long, device=device), act_labels.to(dtype=torch.long, device=device)
            int_outputs, att_outputs, act_outputs = socialc3d(inputs)
            loss_1 = functional.cross_entropy(int_outputs, int_labels)
            loss_2 = functional.cross_entropy(att_outputs, att_labels)
            loss_3 = functional.cross_entropy(act_outputs, act_labels)
            total_loss = loss_1 + loss_2 + loss_3
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        evaluate_model(socialc3d, val_loader, 'validation')

    print("Testing model on test set...")
    evaluate_model(socialc3d, test_loader, 'Test')

    if args.save_weights:
        torch.save(socialc3d.state_dict(), f"{args.save_weights}/socialegonet.pt")
        print(f"Model saved at {args.save_weights}/socialegonet.pt")
