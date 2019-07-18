import torch
from cnns.nnlib.datasets.deeprl.rollouts import get_rollouts_dataset
import os
from cnns.nnlib.pytorch_architecture.get_model_architecture import \
    getModelPyTorch


def pytorch_policy_fn(args):
    models_folder_name = "models"
    models_dir = os.path.join('../nnlib/pytorch_experiments',
                              models_folder_name)
    print("models_dir: ", models_dir)

    train_loader, test_loader = get_rollouts_dataset(args)

    model = getModelPyTorch(args=args)
    # model = torch.nn.DataParallel(model)

    # https://pytorch.org/docs/master/notes/serialization.html
    if args.model_path != "no_model" and args.model_path != "pretrained":
        model.load_state_dict(
            torch.load(os.path.join(models_dir, args.model_path),
                       map_location=args.device))
        msg = "loaded model: " + args.model_path
        # logger.info(msg)
        print(msg)

    model.eval()
    model.to(args.device)

    def infer(ndarray):
        input_tensor = torch.from_numpy(ndarray).to(args.dtype).to(args.device)
        output_tensor = model(input_tensor)
        return output_tensor.to('cpu').detach().numpy()

    return infer
