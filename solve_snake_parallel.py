import numpy as np
import pandas as pd
import os
import sys
import copy
import multiprocessing
from datetime import datetime
import snake_functions as sf

# Change path to script location
# os.chdir(os.path.realpath(sys.argv[0])) # This used to work but not anymore
os.chdir("C:\\Users\\goero\\OneDrive\\Documenten\\Snake-Solver\\")

# Load definition of snake
snake = pd.read_csv("define_snake.txt", sep = "\t", header = 0)
# n_cores = multiprocessing.cpu_count() - 1
n_cores = 2

# Load the cube initialization data and initialize cube + data
# Read in the cube initialisation
init_cube = pd.read_csv("initiate_cube.txt", sep = "\t", header = 0)
# Reform the location part to transform it from a string [x y z] to an array
# that is usable in the code
init_cube['location'] = init_cube.location.str[1:6].replace(' ', '')
init_cube['location'] = init_cube.location.apply(lambda x: np.array(list(map(int, x.split(' ')))))

init_cube = sf.initialize_cube(init_cube, snake)
block_data = init_cube[0]
cube = init_cube[1]

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
def create_queue(cube, block_data, n_cores):
    queue = []
    queue.append(block_data)
    # Create subset that contains all lowest non-forked rows
    while len(queue) < n_cores:
        sub_queue = copy.deepcopy(queue)
        queue = []
        for subq in sub_queue:
            # cube_dummy = sf.update_cube_from_data(cube, subq)[1]
            subq = sf.get_lengths(subq)
            lowest_fork = int(min(subq[subq['len'] > 1].block))
            fork_size = subq[subq.block == lowest_fork].len[0]
            subq = subq.drop(['len'], axis = 1)
            for s in range(fork_size):
                block_data_queue = split_fork(subq, lowest_fork, s)
                cube_dummy = copy.deepcopy(cube)
                cube_dummy = sf.update_cube_from_data(cube_dummy, block_data_queue)
                block_data_queue = sf.continue_path(cube_dummy, block_data_queue)[0]
                queue.append(block_data_queue)
    
    # In many cases, this loop will create more than the required number of elements,
    # keep only the number you need.
    if len(queue) > n_cores:
        queue = queue[0:n_cores]
    
    return(queue)

def solve_cube(cube, block_data):
    
    while len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) > 0 and max_block < 64:
        # This loop will follow any path until it no longer finds any possible directions. When this happens, 
        # the first condition will no longer be valid, as the length of the possible directions will be 0.
        while len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) > 0 and max_block < 64:
            next_step = sf.continue_path(cube_dummy, block_data_dummy)
            block_data_dummy = next_step[0]
            cube_dummy = next_step[1]
            max_block = next_step[2]
        
        # When the inner loop fails, we need to reset the path we took and remove the last direction we took
        # on that path.
        reset = sf.reset_path(cube_dummy, block_data_dummy)
        block_data_dummy = reset[0]
        cube_dummy = reset[1]
        max_block = reset[2]
        # When a the last direction is removed, the length of the possible directions will again be 0.
        # In this case, we want to reset to the last fork that existed, which is the last block where
        # more than one direction existed.
        if len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) == 0 and not max_block == 64:
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
                print("Deepest penetration reached, building back up.")
                highest_penetration = deepest_penetration
            if lowest_fork == highest_penetration:
                highest_penetration += 1
                print(datetime.now())
                print("Went up a level. Highest penetration "+str(highest_penetration))
                
        elif max_block == 64:
            print(datetime.now())
            print("I reached the end.")
            break

#for max_block in range(5,15):
#    next_step = continue_path(cube_dummy, block_data_dummy, max_block)
#    block_data_dummy = next_step[0]
#    cube_dummy = next_step[1]
#    max_block = next_step[2]    
#block = get_max_block(cube_dummy)
#direction_dummy = np.array(block_data_dummy[block_data_dummy.Block == block]["directions"][0][0], dtype = "int64")
#cube_dummy = add_block(cube_dummy, block + 1, get_block_location(cube_dummy, block), direction_dummy)
#location_dummy = get_block_location(cube_dummy, block + 1)
#elbow = (int(snake[snake.Block == (block + 1)]['Elbow']) == 1)
#next_directions = get_possible_directions(cube_dummy, int(block + 1), elbow)
#block_data_dummy = block_data_dummy.append(pd.DataFrame({"Block":int(block + 1),
#                                                         "Elbow":elbow,
#                                                         "location":[location_dummy],
#                                                         "directions":[next_directions]}))

block_data_dummy = copy.deepcopy(block_data)
cube_dummy = copy.deepcopy(cube)
max_block = sf.get_max_block(cube_dummy)
deepest_penetration = int(64)
highest_penetration = int(0)
print('Starting at '+str(datetime.now()))
while len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) > 0 and max_block < 64:
    # This loop will follow any path until it no longer finds any possible directions. When this happens, 
    # the first condition will no longer be valid, as the length of the possible directions will be 0.
    while len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) > 0 and max_block < 64:
        next_step = sf.continue_path(cube_dummy, block_data_dummy)
        block_data_dummy = next_step[0]
        cube_dummy = next_step[1]
        max_block = next_step[2]
    
    block_data_dummy = sf.get_lengths(block_data_dummy)
    if max(block_data_dummy.len) == 1 and max_block < 64:
        print("Couldn't find a solution, your starting positions must have been wrong.")
        # return(block_data_dummy)
        break
    else:
        block_data_dummy = block_data_dummy.drop(['len'], axis = 1)
    # When the inner loop fails, we need to reset the path we took and remove the last direction we took
    # on that path.
    reset = sf.reset_path(cube_dummy, block_data_dummy)
    block_data_dummy = reset[0]
    cube_dummy = reset[1]
    max_block = reset[2]
    # When a the last direction is removed, the length of the possible directions will again be 0.
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
            print("Deepest penetration reached, building back up.")
            highest_penetration = deepest_penetration
        if lowest_fork == highest_penetration:
            highest_penetration += 1
            print(datetime.now())
            print("Went up a level. Highest penetration "+str(highest_penetration))
    elif max_block == 64    :
        print(datetime.now())
        print("I reached the end.")
        break   