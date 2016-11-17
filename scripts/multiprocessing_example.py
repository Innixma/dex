from multiprocessing import Pool
from functools import partial
import time


def wrapper_multi(args):
    return multi_argment_multiply(*args)


def multi_argment_multiply(a, b, c):
    return a*b*c

if __name__ == "__main__":
    size = 10**5

    # http://stackoverflow.com/questions/20713063/writing-into-a-numpy-memmap-still-loads-into-ram-memory

    #### Single Process
    args_together = partial(multi_argment_multiply, b=2, c=3)

    start_time = time.time()
    results = []
    for _ in range(size):
        results.append(multi_argment_multiply(_, 2, 3))
    print("Single Thread Time:\t%f" % (time.time() - start_time))

    #### Multiprocessing
    args_together = partial(multi_argment_multiply, b=2, c=3)

    num_threads = 4

    pool = Pool(num_threads)

    start_time = time.time()
    results_multi = pool.map(args_together, range(size))
    pool.close()
    pool.join()
    print("Multithread Time:\t%f" % (time.time() - start_time))


    #### Multiprocessing
    num_threads = 4

    pool = Pool(num_threads)

    start_time = time.time()
    results_multi2 = pool.map(wrapper_multi, iter((i, 2, 3) for i in range(size)))
    pool.close()
    pool.join()
    print("Multithread2 Time:\t%f" % (time.time() - start_time))

    assert(results == results_multi == results_multi2)
