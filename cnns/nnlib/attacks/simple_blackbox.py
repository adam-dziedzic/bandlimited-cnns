from cnns.nnlib.attacks.simple_blackbox_remote.simba_single import simba_single
from foolbox.attacks.base import Attack
import torch
from cnns.nnlib.attacks.simple_blackbox_remote.torch_01_range import Ranger


class SimbaSingle(Attack):

    def __init__(self, model, args, iterations=10000, epsilon=0.2):
        super(SimbaSingle, self).__init__()
        self.model = model
        self.args = args
        self.dataset = args.dataset
        self.iterations = iterations
        self.epsilon = epsilon
        self.ranger = Ranger(device=args.device)

    def __call__(self, input_or_adv, label=None, unpack=True, unused=None):
        input_tensor = torch.tensor(input_or_adv).unsqueeze(0)
        # input_tensor = self.args.denormalizer(input_tensor.detach())
        input_tensor = self.ranger.to_01(input_tensor.detach(), dataset=self.dataset)
        output_tensor = simba_single(self.model, input_tensor, label,
                            num_iters=self.iterations,
                            epsilon=self.epsilon, dataset=self.dataset)
        # output_tensor = self.args.normalizer(output_tensor.detach())
        output_tensor = self.ranger.to_torch(output_tensor.detach(), dataset=self.dataset)
        return output_tensor.cpu().squeeze().numpy()


if __name__ == "__main__":
    print('SimbaSingle')


