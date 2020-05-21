import os
import torch

from cnns.nnlib.robustness.pni.code.utils_.printing import print_log


def resume_from_checkpoint(net, resume_file, log, optimizer=None, recorder=None, fine_tune=False, start_epoch=0,
                           device=None):
    if os.path.isfile(resume_file):
        print_log("=> loading checkpoint '{}'".format(resume_file), log)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(resume_file, map_location=device)
        if not fine_tune:
            start_epoch = checkpoint['epoch']
            recorder = checkpoint['recorder']
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])

        state_tmp = net.state_dict()
        if 'state_dict' in checkpoint.keys():
            state_tmp.update(checkpoint['state_dict'])
        else:
            state_tmp.update(checkpoint)

        net.load_state_dict(state_tmp)

        print_log("=> loaded checkpoint '{}' (epoch {})".format(
            resume_file, start_epoch), log)
    else:
        msg = "no checkpoint found at '{}'".format(resume_file)
        print_log(msg, log)
        raise Exception(msg)

    return recorder, start_epoch
