from SocialEgoMobile import SocialEgoMobile
from SocialC3D import SocialC3D
from constants import device


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
        pretrained=True,
        dataset=args.dataset,
        modality=config['data']['modality'],
    ).to(device)
    return model
