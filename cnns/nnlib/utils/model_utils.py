import torch
import os
from cnns.nnlib.pytorch_architecture.get_model_architecture import \
    getModelPyTorch
import torchvision.models as models

def load_model(args, pretrained=False):
    model = getModelPyTorch(args=args, pretrained=pretrained)
    # load pretrained weights
    models_folder_name = "models"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(cur_dir, os.path.pardir,
                              "pytorch_experiments", models_folder_name)
    if args.model_path != "no_model":
        model.load_state_dict(
            torch.load(os.path.join(models_dir, args.model_path),
                       map_location=args.device))
        msg = "loaded model: " + args.model_path
        # print(msg)
    return model.eval()

def save_model_from_pretrained():
    model = models.resnet50(pretrained=True)
    model_path = "../pytorch_experiments/models/imagenet_resnet50.model"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    save_model_from_pretrained()