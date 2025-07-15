from constants import device
from tools import load_config, evaluate_model
from Data.Dataset import SocialC3D_Dataset
from torch.utils.data import DataLoader
from Model import create_socialc3d
import argparse


def main():
    parser = argparse.ArgumentParser(description="Test SocialEgoNet on JPL-P4S")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--check_point", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    config = load_config(args.cfg)

    # Load Data
    testset = SocialC3D_Dataset(data_path=config["data"]["path"] + "test/",
                                sequence_length=config["data"]["sequence_length"],
                                dataset=args.dataset, modality=config['data']['modality'])
    test_loader = DataLoader(dataset=testset, batch_size=config["train"]["batch_size"])

    # Load Model
    model = create_socialc3d(args, config)
    model.load_checkpoint(args.check_point)
    model.to(device)

    print("Testing...")
    results = evaluate_model(model, test_loader)

    # Print Results
    for key in ["int", "att", "act"]:
        print(f"{key}_acc: {results[key][1]:.2f}, {key}_f1: {results[key][0]:.2f}")


if __name__ == '__main__':
    main()
