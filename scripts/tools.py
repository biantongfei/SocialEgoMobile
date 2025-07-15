import yaml
import torch
import tqdm
from constants import device
from Data import filter_not_interacting_sample
from sklearn.metrics import f1_score


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


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
