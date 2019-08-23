import foolbox

# Attack the image.
class EmptyAttack(foolbox.attacks.Attack):

    def __call__(self, input_or_adv, label=None, unpack=True, **kwargs):
        return input_or_adv.copy()