"""
Functions for dealing with the speech data input and output.
"""
import numpy as np
import gzip
import logging
from os import path

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
#                            GENERAL I/O FUNCTIONS                            #
# -----------------------------------------------------------------------------#

def smart_open(filename, mode=None):
    """Opens a file normally or using gzip based on the extension."""
    if path.splitext(filename)[-1] == ".gz":
        if mode is None:
            mode = "rb"
        return gzip.open(filename, mode)
    else:
        if mode is None:
            mode = "r"
    return open(filename, mode)


"""
Utility functions.
"""


def swbd_utt_to_label(utt):
    return "_".join(utt.split("_")[:-2])


def swbd_utts_to_labels(utts):
    labels = []
    for utt in utts:
        labels.append(swbd_utt_to_label(utt))
    return labels


"""
Main function to read the data.
"""


def load_swbd_labelled(rng, data_dir, min_count=1):
    """
    Load the Switchboard data with their labels.
    Only tokens that occur at least `min_count` times in the training set
    is considered.
    """

    def get_data_and_labels(set):

        npz_fn = path.join(data_dir, "swbd." + set + ".npz")
        logger.info("Reading: " + npz_fn)

        # Load data and shuffle
        npz = np.load(npz_fn)
        utts = sorted(npz.keys())
        rng.shuffle(utts)
        x = [npz[i] for i in utts]

        # Get labels for each utterance
        labels = swbd_utts_to_labels(utts)

        return x, labels

    train_x, train_labels = get_data_and_labels("train")
    dev_x, dev_labels = get_data_and_labels("dev")
    test_x, test_labels = get_data_and_labels("test")

    logger.info("Finding types with at least " + str(min_count) + " tokens")

    # Determine the types with the minimum count
    type_counts = {}
    for label in train_labels:
        if not label in type_counts:
            type_counts[label] = 0
        type_counts[label] += 1
    min_types = set()
    i_type = 0
    word_to_i_map = {}
    for label in type_counts:
        if type_counts[label] >= min_count:
            min_types.add(label)
            word_to_i_map[label] = i_type
            i_type += 1

    # Filter the sets
    def filter_set(x, labels):
        filtered_x = []
        filtered_i_labels = []
        for cur_x, label in zip(x, labels):
            if label in word_to_i_map:
                filtered_x.append(cur_x)
                filtered_i_labels.append(word_to_i_map[label])
        return filtered_x, filtered_i_labels

    train_x, train_labels = filter_set(train_x, train_labels)
    dev_x, dev_labels = filter_set(dev_x, dev_labels)
    test_x, test_labels = filter_set(test_x, test_labels)

    return [(train_x, train_labels), (dev_x, dev_labels), (test_x, test_labels)]
