import torch
import os
from cnns.nnlib.pytorch_architecture.get_model_architecture import \
    getModelPyTorch


def load_model(args, pretrained=False):
    model = getModelPyTorch(args=args, pretrained=pretrained)
    # load pretrained weights
    models_folder_name = "models"
    models_dir = os.path.join(os.getcwd(), os.path.pardir,
                              "pytorch_experiments", models_folder_name)
    if args.model_path != "no_model":
        model.load_state_dict(
            torch.load(os.path.join(models_dir, args.model_path),
                       map_location=args.device))
        msg = "loaded model: " + args.model_path
        # print(msg)
    return model.eval()
