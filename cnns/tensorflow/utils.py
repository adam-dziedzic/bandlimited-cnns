
def tensor_shape(t):
    return tuple(d.value for d in t.get_shape())

def to_tf(input):
    """
    Transform to tensorflow order of dimensions with channel last.

    :param input: np array
    :return: channels last
    """
    return input.transpose(0, 2, 3, 1)


def from_tf(input):
    """

    :param input: np array
    :return: channels first
    """
    return input.transpose(0, 3, 1, 2)