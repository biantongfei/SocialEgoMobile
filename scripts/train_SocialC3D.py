from Data.Dataset import SocialC3D_Dataset, filter_not_interacting_sample
from Model.SocialC3D import SocialC3D
from constants import device

import argparse
import yaml
import tqdm
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader
from torch.nn import functional


def parse_args():
    parser = argparse.ArgumentParser(description='Train SocialEgoMobile')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--save_weights', type=str, default=None)
    return parser.parse_args()


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


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
        int_outputs, att_outputs, act_outputs = model(inputs)

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
