import time
import numpy as np


# decorator - to time the functions with arguments
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def timeit(statement, number=1):
    t0 = time.time()
    for _ in range(number):
        result = statement()
    t1 = time.time()
    return t1 - t0, result


def timeitrep(statement, number=1, repetition=1):
    """
    Time the execution of the statement `number` of times and repeat it number of `repetitions`.
    The returned timing is the all recorded `repetitions` with discarded potential outliers with the highest and lowest
    times, then averaged. The statement is executed number of times for each repetition and for each repetition we
    record the result from the last run (for a given repetition).

    :param statement: statement to be executed
    :param number: number of runs in each repetitions
    :param repetition: how many time to repeat the experiment
    :return: averge timing (with min, max timings discarded), average value of the results (from each repetition, we
    record the last result, and the last result).
    """
    timings = []
    for _ in range(repetition):
        t0 = time.time()
        statement_result = None
        for _ in range(number):
            statement_result = statement()
        t1 = time.time()
        timings.append(t1 - t0)
        if len(timings) > 2:
            # remove the highest and the lowest time values
            timings.remove(max(timings))
            timings.remove(min(timings))
    return np.average(timings), statement_result