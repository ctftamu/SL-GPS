from multiprocessing import Pool
import time
import math
import sys

print(sys.version)
N = 50000000

def cube(x):
    return math.sqrt(x)

if __name__ == "__main__":
    # first way, using multiprocessing
    start_time = time.time()
    pool = Pool()
    result = pool.map(cube, range(10, N))
    pool.close()
    pool.join()
    finish_time = time.time()
    print("Program finished in {} seconds - using multiprocessing".format(finish_time - start_time))
    print("---")
    # second way, serial computation
    start_time = time.time()
    result = []
    for x in range(10, N):
        result.append(cube(x))
    finish_time = time.time()
    print("Program finished in {} seconds".format(finish_time - start_time))

