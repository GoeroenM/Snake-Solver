import numpy as np
import pandas as pd
import os
import copy
from multiprocessing.pool import Pool
from functools import partial
from datetime import datetime
# Change path to script location
# os.chdir(os.path.realpath(sys.argv[0])) # This used to work but not anymore
os.chdir("C:\\Users\\goero\\OneDrive\\Documenten\\Snake-Solver\\")
import snake_functions as sf

# Function that splits a fork and returns the block data up until, and including
# the fork, but only one of the fork paths.
def split_fork(block_data, fork, path):
    block_data_sub = block_data[0:(fork - 1)]
    data_split = block_data[block_data.block == fork]
    new_direction = np.zeros((1,3))
    new_direction = new_direction + data_split.directions[0][path]
    fork_split = pd.DataFrame({"block":data_split.block,
                                    "elbow":data_split.elbow, 
                                    "location":data_split.location,
                                    "directions":[new_direction]})
    block_data_sub = block_data_sub.append(fork_split)
    return(block_data_sub)

        
# Function that creates a queu of objects to run parallel process on.
# Create the queue starting with the lowest fork and then moving upwards until
# we've reached the number of CPU cores.
def create_queue(cube, block_data, n_cores, snake):
    queue = []
    queue.append(block_data)
    # Create subset that contains all lowest non-forked rows
    while len(queue) < n_cores:
        sub_queue = copy.deepcopy(queue)
        queue = []
        for subq in sub_queue:
            # cube_dummy = sf.update_cube_from_data(cube, subq)[1]
            subq = sf.get_number_of_directions(subq)
            # If all directions have length 1, the lowest fork is going to 
            # be the max block
            if max(subq['len']) == 1:
                lowest_fork = int(max(subq['block']))
            else:
                lowest_fork = int(min(subq[subq['len'] > 1].block))
            fork_size = subq[subq.block == lowest_fork].len[0]
            subq = subq.drop(['len'], axis = 1)
            for s in range(fork_size):
                block_data_queue = split_fork(subq, lowest_fork, s)
                cube_dummy = copy.deepcopy(cube)
                cube_dummy = sf.update_cube_from_data(cube_dummy, block_data_queue)
                block_data_queue = sf.continue_path(cube_dummy, snake, block_data_queue)[0]
                queue.append(block_data_queue)
    
    # In many cases, this loop will create more than the required number of elements,
    # keep only the number you need.
    if len(queue) > n_cores:
        queue = queue[0:n_cores]
    
    return(queue)

def solve_cube(cube, block_data, snake):
    block_data_dummy = copy.deepcopy(block_data)
    cube_dummy = copy.deepcopy(cube)
    max_block = sf.get_max_block(cube_dummy)
    deepest_penetration = int(64)
    highest_penetration = int(0)
    print('Starting at '+str(datetime.now()))
    start_time = datetime.now()
    while len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) > 0 and max_block < 64:
        # This loop will follow any path by always taking the first path until it no longer finds any possible directions.
        # When this happens, the first condition will no longer be valid, as the length of the possible directions 
        # will be 0 and we need to reset the path.
        while len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) > 0 and max_block < 64:
            next_step = sf.continue_path(cube = cube_dummy, block_data = block_data_dummy, snake = snake)
            block_data_dummy = next_step[0]
            cube_dummy = next_step[1]
            max_block = next_step[2]
        
        # Before resetting the path, check if the cube is not finished or a complete dead end.
        block_data_dummy = sf.get_number_of_directions(block_data_dummy)
        if max(block_data_dummy.len) == 1 and max_block < 64:
            print("Couldn't find a solution, your starting positions must have been wrong.")
            return(block_data_dummy, cube_dummy)
            stop_time = datetime.now()
            # break
        elif max_block == 64:
            stop_time = datetime.now()
            print(stop_time)
            print("Cube solved.")
            print("Time elapsed: "+str(stop_time - start_time))
            # Now clean up the cube data
            block_data = block_data_dummy[['block', 'location', 'elbow']]
            cube = cube_dummy
            return(block_data, cube)
            # break
        else:
            block_data_dummy = block_data_dummy.drop(['len'], axis = 1)
        # When the inner loop fails, we need to reset the path we took and remove the last direction we took
        # on that path.
        reset = sf.reset_path(cube_dummy, block_data_dummy)
        block_data_dummy = reset[0]
        cube_dummy = reset[1]
        max_block = reset[2]
        # When the last direction is removed, the length of the possible directions will again be 0.
        # In this case, we want to reset to the last fork that existed, which is the last block where
        # more than one direction existed.
        if len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) == 0 and max_block < 64:
            reset = sf.reset_to_latest_fork(cube_dummy, block_data_dummy)
            block_data_dummy = reset[0]
            cube_dummy = reset[1]
            max_block = reset[2]
            lowest_fork = reset[3]
            if max_block < deepest_penetration:
                deepest_penetration = max_block
                print(datetime.now())
                print("Resetting to fork. Lowest fork "+str(lowest_fork)+". Deepest penetration "+str(deepest_penetration))
            if max_block == lowest_fork:
                print(datetime.now())
                print("Deepest penetration reached, building back up. Lowest fork "+str(lowest_fork))
                highest_penetration = deepest_penetration
            if lowest_fork == highest_penetration:
                highest_penetration += 1
                print(datetime.now())
                print("Went up a level. Highest penetration "+str(highest_penetration))

def main():
    # Load definition of snake
    snake = pd.read_csv("define_snake.txt", sep = "\t", header = 0)
    
    # Pick however many cores you want to use. Either based on your PC or an absolute number of your choosing.
    # n_cores = multiprocessing.cpu_count() - 1
    n_cores = 2
    
    # Load the cube initialization data and initialize cube + data
    # Read in the cube initialisation
    init_cube = pd.read_csv("initiate_cube_reverse.txt", sep = "\t", header = 0)
    # Reform the location part to transform it from a string [x y z] to an array
    # that is usable in the code
    init_cube['location'] = init_cube.location.str[1:6].replace(' ', '')
    init_cube['location'] = init_cube.location.apply(lambda x: np.array(list(map(int, x.split(' ')))))
    
    init_cube = sf.initialize_cube(init_cube, snake)
    block_data = init_cube[0]
    cube = init_cube[1]
    queue = create_queue(cube, block_data, n_cores, snake)
    solve_q = partial(solve_cube, cube, snake)
    with Pool(2) as p:
        p.map(solve_q, queue)
        p.close()
        p.join()
        