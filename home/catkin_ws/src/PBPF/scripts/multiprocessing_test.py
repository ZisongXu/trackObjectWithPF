#!/usr/bin/python3
import time
import multiprocessing
import numpy as np
import random

num_of_particles = 50


def main_non_parellised():
    results = []
    your_parameter = 10
    start = time.time()
    for i in range(num_of_particles):
        result = function_to_parallelise(your_parameter, None)
        results.append(result)

    end = time.time()
    print(end - start)

    return results


def main_parallelised():
    # Divided by 2 because in most systems there is hyper-threading, meaning
    # that you get 2x your actual cores, but they are threads, hence true
    # parallelisation should happen on the actual cores (i.e., cpu_count / 2).
    num_cpus = int(multiprocessing.cpu_count() / 2)

    particles_per_cpu = num_of_particles // num_cpus
    leftover_particles = num_of_particles - (particles_per_cpu * num_cpus)

    num_cores = [num_cpus] * particles_per_cpu + [leftover_particles]

    results = []
    processes = []

    start = time.time()
    for num in num_cores:
        for i in range(num):
            # Use pipe if you need processes to return some result, remove if not.
            pipe_parent, pipe_child = multiprocessing.Pipe()
            process = multiprocessing.Process(target=function_to_parallelise, args=(your_argument, pipe_child))
            process.start()
            processes.append((process, pipe_parent))

        for process, pipe in processes:
            process.join()
            results.append(pipe.recv())

    end = time.time()
    print(end - start)

    return results


def function_to_parallelise(your_parameter, pipe):
    # If you use random values in the parallelised function, you need to
    # re-generate the seed otherwise each process will generate the same seed.

    # If you use numpy random library
    a = np.random.seed()

    # If you use python's random library
    b = random.seed()


    # Do your simulation here...


    if pipe:
        # Send result back if you want to (can't return, use pipe).
        pipe.send(a)
result = main_non_parellised()
print("result:",result)
