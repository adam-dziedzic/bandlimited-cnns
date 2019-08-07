from matplotlib import pyplot as plt
import numpy as np


def plot_hist(data, title=''):
    n, bins, patches = plt.hist(
        x=data, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    # plt.text(23, 45, r"$\mu=15, b=3$")
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()


if __name__ == "__main__":
    d = np.random.laplace(loc=15, scale=3, size=500)
    plot_hist(d)