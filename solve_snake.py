import numpy as np
import pandas as pd
import os
import sys
import copy
from datetime import datetime
import snake_functions as sf

# Change path to script location
# os.chdir(os.path.realpath(sys.argv[0])) # This used to work but not anymore
os.chdir("C:\\Users\\goero\\OneDrive\\Documenten\\Snake-Solver\\")

# Load definition of snake
snake = pd.read_csv("define_snake.txt", sep = "\t", header = 0)

# Read in the cube initialisation
init_cube = pd.read_csv("initiate_cube.txt", sep = "\t", header = 0)
# Reform the location part to transform it from a string [x y z] to an array
# that is usable in the code
init_cube['location'] = init_cube.location.str[1:6].replace(' ', '')
init_cube['location'] = init_cube.location.apply(lambda x: np.array(list(map(int, x.split(' ')))))

init_cube = sf.initialize_cube(init_cube, snake)
block_data = init_cube[0]
cube = init_cube[1]

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
    elif max_block == 64:
        print(datetime.now())
        print("I reached the end.")
        break


for max_block in range(5,15):
    next_step = continue_path(cube_dummy, block_data_dummy, max_block)
    block_data_dummy = next_step[0]
    cube_dummy = next_step[1]
    max_block = next_step[2]    
block = get_max_block(cube_dummy)
direction_dummy = np.array(block_data_dummy[block_data_dummy.Block == block]["directions"][0][0], dtype = "int64")
cube_dummy = add_block(cube_0dummy, block + 1, get_block_location(cube_dummy, block), direction_dummy)
location_dummy = get_block_location(cube_dummy, block + 1)
elbow = (int(snake[snake.Block == (block + 1)]['Elbow']) == 1)
next_directions = get_possible_directions(cube_dummy, int(block + 1), elbow)
block_data_dummy = block_data_dummy.append(pd.DataFrame({"Block":int(block + 1),
                                                         "Elbow":elbow,
                                                         "location":[location_dummy],
                                                         "directions":[next_directions]}))

    