from cnns.nnlib.utils.exec_args import get_args
from cnns.deeprl.pytorch_model import pytorch_policy_fn
from cnns.deeprl.models import run_model
from cnns.nnlib.utils.general_utils import PolicyType

if __name__ == '__main__':
    args = get_args()

    assert args.policy_type == PolicyType.PYTORCH_BEHAVE

    # for rollouts in [100]:
    #     args.rollouts = rollouts
    #     run_model(args=args, policy_fn=policy_fn)

    # args.rollouts = 100
    # run_model(args=args, policy_fn=policy_fn)

    # for rollouts in [1500, 2500, 3000, 3500, 4000, 4500, 5000]:
    for rollouts in [100]:
        args.rollouts = rollouts
        # for env_name in ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Reacher-v2']:
        # for env_name in ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Reacher-v2']:
        for env_name in ['Ant-v2']:
            args.env_name = env_name
            for learn_policy_file in ['Ant-v2-10.model',
                                      'Ant-v2-100.model',
                                      'Ant-v2-1000-epoch-64.model',
                                      'Ant-v2-1000-epoch-90.model',
                                      'Ant-v2-1500-epoch-34.model']:
                args.learn_policy_vile = 'behave_models/' + learn_policy_file
                policy_fn = pytorch_policy_fn(args=args)
                args.rollout_file = 'expert_data/' + args.learn_policy_file.replace(
                    '/', '-') + '-' + str(rollouts) + '.pkl'
                print(args.learn_policy_file)
                run_model(args=args, policy_fn=policy_fn)
