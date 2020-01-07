import foolbox

from cnns.nnlib.robustness.pytorch_model import get_model

def get_fmodel(args):
    pytorch_model = get_model(args)
    fmodel = foolbox.models.PyTorchModel(
        pytorch_model,
        bounds=(args.min, args.max),
        num_classes=args.num_classes)
    return fmodel, pytorch_model, args.from_class_idx_to_label