import numpy as np
import pandas as pd
import os
import sys
import copy
import multiprocessing
import snake_functions as sf

# Change path to script location
# os.chdir(os.path.realpath(sys.argv[0])) # This used to work but not anymore
os.chdir("C:\\Users\\goero\\OneDrive\\Documenten\\Snake-Solver\\")

# Load definition of snake
snake = pd.read_csv("define_snake.txt", sep = "\t", header = 0)
n_cores = multiprocessing.cpu_count()
n_cores = 2

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
    block_data = sf.get_lengths(block_data)
    lowest_fork = int(min(block_data[block_data['len'] > 1].block))
    fork_size = block_data[block_data.block == lowest_fork].len[0]
    block_data = block_data.drop(['len'], axis = 1)
    # Create subset that contains all lowest non-forked rows
    block_data_sub = block_data[0:(lowest_fork - 1)]
    while len(queue) < n_cores:
        for q in queue:
            q = sf.get_lengths(q)
            lowest_fork = int(min(q[q['len'] > 1].block))
            fork_size = q[q.block == lowest_fork].len[0]
            q = q.drop(['len'])
            for s in range(fork_size):
                block_data_queue = split_fork(block_data, lowest_fork, s)
                queue.append(block_data_queue)
            for q in range(len(queue)):
                max_block = sf.get_max_block(cube)
                new_direction = queue[q][queue[q].block == max_block].directions[0][0]
                cube_dummy = sf.add_block(cube, block = max_block + 1, \
                                direction = new_direction)
                queue[q] = sf.continue_path(cube, queue[q])[0]
                queue[q] = sf.get_lengths(queue[q])
                
# Function that creates a queu of objects to run parallel process on.
# Create the queue starting with the lowest fork and then moving upwards until
# we've reached the number of CPU cores.
def create_queue(cube, block_data, n_cores):
    block_data = sf.get_lengths(block_data)
    lowest_fork = int(min(block_data[block_data['len'] > 1].block))
    fork_size = block_data[block_data.block == lowest_fork].len[0]
    block_data = block_data.drop(['len'], axis = 1)
    # Create subset that contains all lowest non-forked rows
    # block_data_sub = block_data[0:(lowest_fork - 1)]
    sub_queue = []
    for s in range(fork_size):
        block_data_queue = split_fork(block_data, lowest_fork, s)
        sub_queue.append(block_data_queue)
    
    queue = []
    for q in range(len(sub_queue)):
        sub_queue[q] = sf.continue_path(cube, sub_queue[q])[0]
        sub_queue[q] = sf.get_lengths(sub_queue[q])
        max_block = int(max(sub_queue[q].block[0]))        
        new_direction = sub_queue[q][sub_queue[q].block == max_block - 1].directions[0][0]
        cube_dummy = copy.deepcopy(cube)
        cube_dummy = sf.add_block(cube_dummy, block = max_block, \
                                direction = new_direction)
        lowest_fork = int(min(sub_queue[q][sub_queue[q]['len'] > 1].block))
        fork_size = sub_queue[q][sub_queue[q].block == lowest_fork].len[0]
        sub_queue[q] = sub_queue[q].drop(['len'], axis = 1)
        for s in range(fork_size):
            block_data_queue = split_fork(sub_queue[q], lowest_fork, s)
            queue.append(block_data_queue)
    
    if len(queue) > n_cores:
        queue = queue[0:n_cores]
    
    return(queue)
        
block_data_dummy = copy.deepcopy(sub_queue[q])
lowest_fork = int(min(queue[q][queue[q]['len'] > 1].block))
cube = sf.remove_block(cube, 6)
   
# Initialize cube with known positions
cube = np.zeros((4,4,4))
cube[3][1][0] = int(1)
cube[3][0][0] = int(2)
cube[2][0][0] = int(3)
cube[1][0][0] = int(4)
cube[0][0][0] = int(5)

max_block = sf.get_max_block(cube)
last_direction = sf.get_last_direction(cube)

# Initialize block metadata
columns = ["block", "elbow", "location", "directions"]
block_data = pd.DataFrame(columns = columns)

for block in np.unique(cube)[np.unique(cube) != 0]:
    block_number = block
    block_location = sf.get_block_location(cube, block_number)
    elbow = (int(snake[snake.block == max_block]['elbow']) == 1)
    if block_number < max_block:
        direction = np.zeros((1,3))
        direction = direction + sf.get_direction(cube, block_number, (block_number + 1))
    else:
        # Set the elbow variable as boolean, rather than 0/1
        elbow = (int(snake[snake.block == max_block]['elbow']) == 1)
        direction = sf.get_possible_directions(cube, block_number, elbow)
    block_data = block_data.append(pd.DataFrame({"block":block_number,
                                    "elbow":elbow, 
                                    "location":[block_location],
                                    "directions":[direction]}))
    
# blocks_to_go = np.setdiff1d(snake["Block"], np.unique(cube))
blocks_to_go = snake[max_block:]

block_data_dummy = copy.deepcopy(block_data)
cube_dummy = copy.deepcopy(cube)
# This bottom one is redundant, as it's already called higher up, I just
# put it here for rerunning coe after testing.
max_block = sf.get_max_block(cube)
# This merely exists for progress reporting
deepest_penetration = int(64) 
while len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) > 0:
    while len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) > 0:
        next_step = sf.continue_path(cube_dummy, block_data_dummy)
        block_data_dummy = next_step[0]
        cube_dummy = next_step[1]
        max_block = next_step[2]

    reset = sf.reset_path(cube_dummy, block_data_dummy)
    block_data_dummy = reset[0]
    cube_dummy = reset[1]
    max_block = reset[2]
    if len(block_data_dummy[block_data_dummy.block == max_block]["directions"][0]) == 0 and not max_block == 64:
        reset = sf.reset_to_latest_fork(cube_dummy, block_data_dummy)
        block_data_dummy = reset[0]
        cube_dummy = reset[1]
        max_block = reset[2]
        if max_block < deepest_penetration:
            deepest_penetration = max_block
            print("Resetting to fork. Lowest fork "+str(reset[3])+". Deepest penetration "+str(deepest_penetration))
    elif max_block == 64:
        print("I reached the end.")
        break

for max_block in range(5,15):
    next_step = continue_path(cube_dummy, block_data_dummy, max_block)
    block_data_dummy = next_step[0]
    cube_dummy = next_step[1]
    max_block = next_step[2]    
block = get_max_block(cube_dummy)
direction_dummy = np.array(block_data_dummy[block_data_dummy.Block == block]["directions"][0][0], dtype = "int64")
cube_dummy = add_block(cube_dummy, block + 1, get_block_location(cube_dummy, block), direction_dummy)
location_dummy = get_block_location(cube_dummy, block + 1)
elbow = (int(snake[snake.Block == (block + 1)]['Elbow']) == 1)
next_directions = get_possible_directions(cube_dummy, int(block + 1), elbow)
block_data_dummy = block_data_dummy.append(pd.DataFrame({"Block":int(block + 1),
                                                         "Elbow":elbow,
                                                         "location":[location_dummy],
                                                         "directions":[next_directions]}))

    