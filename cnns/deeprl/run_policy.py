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
    elif args.policy_type == PolicyType.PYTORCH_DAGGER:
        policy_fn = pytorch_policy_fn(args=args)
    else:
        raise Exception(f'Unknown policy type: {args.policy_type.name}')
    print('loaded and built')

    # for rollouts in [100]:
    #     args.rollouts = rollouts
    #     run_model(args=args, policy_fn=policy_fn)

    # args.rollouts = 100
    # run_model(args=args, policy_fn=policy_fn)

    for rollouts in [1500, 2500, 3000, 3500, 4000, 4500, 5000]:
        args.rollouts = rollouts
        # for env_name in ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Reacher-v2']:
        for env_name in ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Reacher-v2']:
            args.env_name = env_name
            if args.policy_type == PolicyType.EXPERT:
                args.expert_policy_file = "experts/" + args.env_name + ".pkl"
                policy_fn = load_policy(args.expert_policy_file)
            args.rollout_file = 'expert_data/' + args.env_name + '-' + str(rollouts) + '.pkl'
            run_model(args=args, policy_fn=policy_fn)
