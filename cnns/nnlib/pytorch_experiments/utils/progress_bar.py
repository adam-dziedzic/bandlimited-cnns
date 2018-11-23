"""
From: https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
"""

import time
import os
import sys

try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except:
    term_width = 120

TOTAL_BAR_LENGTH = 40.
is_begining = True
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, epoch=None, msg=None):
    global last_time, begin_time, is_begining, TOTAL_BAR_LENGTH

    text_len = 0

    if epoch:
        str = f"Epoch: {epoch}"
        text_len += len(str)
        sys.stdout.write(str)

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append(' Step time: %s' % format_time(step_time))
    L.append(' | Tot time: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)

    text_len += len(msg) + 10
    if is_begining:
        # Freeze the Total bar length.
        TOTAL_BAR_LENGTH = max(20, term_width - text_len)
        is_begining = False

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    sys.stdout.write(msg)

    # Erase to the beginning of the line.
    for i in range(text_len + int(TOTAL_BAR_LENGTH) + 3):
        sys.stdout.write('\b')

    if current >= total - 1:
        sys.stdout.write('\n')
        begin_time = time.time()  # Reset for new bar.
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
