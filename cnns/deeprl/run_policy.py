from cnns.nnlib.utils.exec_args import get_args
from cnns.deeprl.load_policy import load_policy
from cnns.deeprl.pytorch_model import pytorch_policy_fn
from cnns.deeprl import behavioral_cloning
from cnns.deeprl.models import run_model
from cnns.nnlib.utils.general_utils import PolicyType


if __name__ == '__main__':
    args = get_args()

    print('loading and building expert policy')
    if args.policy_type == PolicyType.TENSORFLOW_BEHAVE:
        args.hidden_units = 100
        policy_fn = behavioral_cloning.load_policy(args=args)
    elif args.policy_type == PolicyType.PYTORCH_BEHAVE:
        policy_fn = pytorch_policy_fn(args=args)
    elif args.policy_type == PolicyType.EXPERT:
        policy_fn = load_policy(args.expert_policy_file)
    else:
        raise Exception(f'Unknown policy type: {args.policy_type.name}')
    print('loaded and built')
    args.num_rollouts = 100
    run_model(args=args, policy_fn=policy_fn)
