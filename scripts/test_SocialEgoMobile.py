from constants import device
from tools import load_config, evaluate_model
from Data.Dataset import Pose_Dataset
from Data.PoseDataLoader import Pose_DataLoader
from Model import create_socialegomobile
import argparse


def main():
    parser = argparse.ArgumentParser(description="Test SocialEgoNet on JPL-P4S")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--check_point", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    config = load_config(args.cfg)

    # Load Data
    testset = Pose_Dataset(config["data"]["path"] + "test/", config["data"]["sequence_length"])
    test_loader = Pose_DataLoader(dataset=testset, sequence_length=config["data"]["sequence_length"],
                                  batch_size=config["train"]["batch_size"],
                                  keypoint_drop_rate=config['data']['keypoint_drop_rate'],
                                  frame_drop_rate=config['data']['frame_drop_rate'])

    # Load Model
    model = create_socialegomobile(args, config)
    model.load_checkpoint(args.check_point)
    model.to(device)

    print("Testing...")
    results = evaluate_model(model, test_loader)

    # Print Result
    for key in ["int", "att", "act"]:
        print(f"{key}_acc: {results[key][1]:.2f}, {key}_f1: {results[key][0]:.2f}")


if __name__ == '__main__':
    main()
