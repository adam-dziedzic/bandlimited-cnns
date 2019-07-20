import torch
from cnns.nnlib.datasets.deeprl.rollouts import get_rollouts_dataset
from cnns.nnlib.pytorch_architecture.get_model_architecture import \
    getModelPyTorch


def load_model(args):
    # Set the params of the model.
    get_rollouts_dataset(args)

    model = getModelPyTorch(args=args)
    # model = torch.nn.DataParallel(model)

    model.to(args.device)

    if args.learn_policy_file != 'no_policy_file':
        model.load_state_dict(
            torch.load(args.learn_policy_file, map_location=args.device))
        msg = "loaded model: " + args.learn_policy_file
        print(msg)

    return model


def pytorch_policy_fn(args, model=None):
    if model is None:
        model = load_model(args=args)
    model.eval()
    def infer(ndarray):
        input_tensor = torch.from_numpy(ndarray).to(args.dtype).to(args.device)
        output_tensor = model(input_tensor)
        return output_tensor.to('cpu').detach().numpy()

    return infer
