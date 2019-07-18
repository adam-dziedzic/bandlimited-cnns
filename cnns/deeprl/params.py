# Select the environment
# env_name = "Ant-v2"
# env_name = "HalfCheetah-v2"
# env_name = "Hopper-v2" # 2nd best visualization but still a bit too fast
# and going out of the checkerboard
# env_name = "Humanoid-v2"


class Arguments(object):
    """
    Encapsulate all the arguments for the running program and carry them through
    the execution.
    """

    def __init__(self):
        self.env_name = "Reacher-v2"
        self.expert_data_dir = 'expert_data/'
        self.behave_model_prefix = 'behave_models/'
        self.dagger_model_prefix = 'dagger_models/'
        self.input_output_size = {}
        self.hidden_units = 64
        # train_steps = 1000000
        self.train_steps = 1000000
        self.rollouts = 100000
        self.verbose = False
        self.max_timesteps = None
        self.render = True
        self.model_type = 'expert'  # 'student_behave' or 'student_dagger' or 'expert'
        self.expert_policy_file = self.get_model_file()

    def get_model_file(self):
        if self.verbose:
            print('model file rollouts: ', self.rollouts)
            print('model file train steps: ', self.train_steps)

        if self.model_type == 'student_behave':
            model_file = self.behave_model_prefix + self.env_name + '-rollouts' + str(
            self.rollouts) + '-train-steps-' + str(self.train_steps) + '.ckpt'
        elif self.model_type == 'expert':
            model_file = "experts/" + self.env_name + ".pkl"
        else:
            raise Exception(f'Unknown model type: {self.model_type}')
        return model_file

    def get_str(self):
        args_dict = self.__dict__
        args_str = " ".join(
            ["--" + str(key) + "=" + str(value) for key, value in
             sorted(args_dict.items())])
        return args_str

    def set_parsed_args(self, parsed_args):
        # Make sure you do not miss any properties.
        # https://stackoverflow.com/questions/243836/how-to-copy-all-properties-of-an-object-to-another-object-in-python
        parsed_dict = parsed_args.__dict__.copy()
        for arg in parsed_dict.keys():
            self.__dict__[arg] = parsed_dict[arg]
