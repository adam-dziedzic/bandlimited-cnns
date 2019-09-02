import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

imagenet_example_label = 'hammerhead, hammerhead shark'
imagenet_example_id = 4

imagenet_example = np.load(dir_path + '/imagenet_249.npy')

if __name__ == "__main__":
    img = imagenet_example
    print('img size: ', img.size)
    print('img shape: ', img.shape)
    print('img type and dtype: ', type(img), ',', img.dtype)
