import numpy as np
import pandas as pd
import os
import copy
import multiprocessing

# Change path to script location
# os.chdir(os.path.realpath(sys.argv[0])) # This used to work but not anymore
os.chdir("C:\\Users\\goero\\OneDrive\\Documenten\\Snake-Solver\\")

# Load definition of snake
snake = pd.read_csv("define_snake.txt", sep = "\t", header = 0)
n_cores = multiprocessing.cpu_count()

# Initialize cube based on input data.
# Input format needs to the following/
# 2 columns named 'block' and 'location' where location is an array with the
# location in x, y, z coordinates.
def initialize_cube(init_cube, snake, cube_size = 4):
    cube = np.zeros((cube_size, cube_size, cube_size))
    for i in init_cube.block:
        location = init_cube[init_cube.block == i].location[(i - 1)]
        if cube[location[0]][location[1]][location[2]] != 0:
            print("You tried to put two blocks in the same location.\n")
            print("Aborting cube initilisation, please fill in correct locations.")
            cube = np.zeros((cube_size, cube_size, cube_size))
            break
        else:
            cube[location[0]][location[1]][location[2]] = int(i)
    
    # Now initialize the block data
    max_block = get_max_block(cube)
    columns = ["block", "elbow", "location", "directions"]
    block_data = pd.DataFrame(columns = columns)
    if max_block == 0:
        print("Your cube is empty, so is your block_data.")
    else:
        for block in np.unique(cube)[np.unique(cube) != 0]:
            block_number = block
            block_location = get_block_location(cube, block_number)
            elbow = (int(snake[snake.block == block_number]['elbow']) == 1)
            if block_number < max_block:
                direction = np.zeros((1,3))
                direction = direction + get_direction(cube, block_number, (block_number + 1))
            else:
                # Set the elbow variable as boolean, rather than 0/1
                elbow = (int(snake[snake.block == max_block]['elbow']) == 1)
                direction = get_possible_directions(cube, block_number, elbow)
            block_data = block_data.append(pd.DataFrame({"block":block_number,
                                    "elbow":elbow, 
                                    "location":[block_location],
                                    "directions":[direction]}))
    
    return(block_data, cube)

def update_cube_from_data(cube, block_data, verbose = False):
    max_block_cube = get_max_block(cube)
    max_block_data = int(max(block_data.block))
    if max_block_cube == max_block_data:
        if verbose:
            print("Cube and data are of equal size, I have nothing to do.")
        return(cube)
    elif max_block_cube > max_block_data:
        if verbose:
            print("Your cube is bigger than your data, something went wrong.")
        return(cube)
    else:
        cube_dummy = copy.deepcopy(cube)
        for b in range(max_block_cube, max_block_data):
            direction = block_data[block_data.block == b]['directions'][0][0]
            cube_dummy = add_block(cube, b + 1, direction)
        return(cube_dummy)

# Some practical functions
def add_block(cube, block, direction):
    from_location = get_block_location(cube, (block - 1))
    new_position = np.array(from_location) + np.array(direction)
    # Check if the new position is within the bounds of the cube
    if any(x < 0 for x in new_position) or any(x > 3 for x in new_position):
        raise ValueError("Block out of bounds.")
    # Check if the new position is already taken
    elif cube[int(new_position[0])][int(new_position[1])][int(new_position[2])] != 0:
        raise ValueError("Position already taken.")
    else:
        cube[int(new_position[0])][int(new_position[1])][int(new_position[2])] = block
    return(cube)

def get_block_number(cube, position):
    position = np.array(position)
    return cube[position[0]][position[1]][position[2]]

def get_max_block(cube):
    return(int(max(np.unique(cube))))

def get_block_location(cube, block_number):
    return(np.array(np.where(cube == block_number)).flatten())

def remove_block(cube, block):
    location = get_block_location(cube, block)
    cube[location[0]][location[1]][location[2]] = 0
    return(cube)

def get_direction(cube, block1, block2):
    location1 = get_block_location(cube, block1)
    location2 = get_block_location(cube, block2)
    return(location2 - location1)

def get_last_direction(cube):
    max_ = get_max_block(cube)
    second = max_ - 1
    max_location = get_block_location(cube, max_)
    second_location = get_block_location(cube, second)
    return(max_location - second_location)

def get_possible_directions(cube, block, elbow):
    location = get_block_location(cube, block)
    last_direction = get_direction(cube, (block - 1), block)
    if elbow:
        last_axis = np.where(last_direction != 0)[0]
        other_axes = np.where(last_direction == 0)[0]
        # Create new possible directions based on the last axis of rotation
        # Initiate empty array
        possible_directions = np.zeros((4,3))
        dummy = 0
        # For both axes, need to create 2 directions, the previous axis is 0
        for axis in other_axes:
            for x in range(2):
                possible_directions[dummy][last_axis] = 0
                possible_directions[dummy][axis] = (-1)**x
                dummy = dummy + 1
        
        # If a direction is out of bounds or the position is already taken,
        # we'll remove those possible directions.
        remove_idx = []
        for d in range(len(possible_directions)):
            new_position = location.flatten() + np.array(possible_directions[d], dtype = 'int64')
            if any(x < 0 for x in new_position) or any(x > 3 for x in new_position):
                remove_idx.append(d)
                continue
            elif cube[new_position[0]][new_position[1]][new_position[2]] != 0:
                remove_idx.append(d)
                continue
        
        # Keep only possible directions
        keep_idx = [i not in remove_idx for i in [0,1,2,3]]
        possible_directions = possible_directions[keep_idx]    
                
    else:
        #The reason I predefine possible directions here and sum rather than
        #just taking get_last_direction is to ensure that in both the if and
        #the else, possible_directions is returned in the same format.
        possible_directions = np.zeros((1,3))
        possible_directions = possible_directions + get_last_direction(cube)
    return(possible_directions)

def continue_path(cube, block_data, snake):
    block_data_dummy = copy.deepcopy(block_data)
    cube_dummy = copy.deepcopy(cube)
    block = get_max_block(cube_dummy)
    direction_dummy = np.array(block_data_dummy[block_data_dummy.block == block]["directions"][0][0], dtype = "int64")
    try:
        cube_dummy = add_block(cube_dummy, block + 1, direction_dummy)
        location_dummy = get_block_location(cube_dummy, block + 1)
        elbow = (int(snake[snake.block == (block + 1)]['elbow']) == 1)
        next_directions = get_possible_directions(cube_dummy, int(block + 1), elbow)
        block_data_dummy = block_data_dummy.append(pd.DataFrame({"block":int(block + 1),
                                                         "elbow":elbow,
                                                         "location":[location_dummy],
                                                         "directions":[next_directions]}))
        max_block = get_max_block(cube_dummy)
    except ValueError:
        # A ValueError is only going to happen when you're trying to continue
        # on from one elbow to the next. So in this case, we'll have to go back
        # to the last elbow instead of just remove the last block and try a new
        # direction.
        last_elbow = int(max(block_data_dummy[block_data_dummy.elbow]["block"]))
        for b in range(last_elbow, block):
            cube_dummy = remove_block(cube_dummy, (b + 1))
        
        # Remove last direction and update block_data_dummy
        old_data = block_data_dummy[block_data_dummy.block == last_elbow]
        replacement_data = pd.DataFrame({'block':old_data.block,
                                         'elbow':old_data.elbow,
                                         'location':old_data.location,
                                         'directions':[old_data.directions[0][1:]]})
        block_data_dummy = block_data_dummy[0:(last_elbow - 1)]
        block_data_dummy = block_data_dummy.append(replacement_data)
        max_block = get_max_block(cube_dummy)
    return(block_data_dummy, cube_dummy, max_block)

# This function is run when the continue_path function finds a dead end.
# Fix the cube by removing the last block and removing the last taken path.    
def reset_path(cube, block_data):
    max_block = get_max_block(cube)
    cube = remove_block(cube, max_block)
    old_data = block_data[block_data.block == (max_block - 1)]
    # Take all the same data, but remove the last taken direction from the
    # previous block
    replacement_data = pd.DataFrame({'block':old_data.block,
                                         'elbow':old_data.elbow,
                                         'location':old_data.location,
                                         'directions':[old_data.directions[0][1:]]})
    block_data = block_data[0:(max_block - 2)]
    block_data = block_data.append(replacement_data)
    max_block = get_max_block(cube)
    return(block_data, cube, max_block)

def get_lengths(block_data):
    block_data['len'] = int()
    for i in range(0,len(block_data)):
        block_data.iat[i, 4] = len(block_data[block_data.block == i + 1]["directions"][0])
    return(block_data)
        

def reset_to_latest_fork(cube, block_data):
    max_block = get_max_block(cube)
    block_data = get_lengths(block_data)
    latest_fork = int(max(block_data[block_data['len'] > 1].block))
    lowest_fork = int(min(block_data[block_data['len'] > 1].block))
    block_data = block_data.drop(['len'], axis = 1)
    old_data = block_data[block_data.block == latest_fork]
    replacement_data = pd.DataFrame({'block':old_data.block,
                                         'elbow':old_data.elbow,
                                         'location':old_data.location,
                                         'directions':[old_data.directions[0][1:]]})
    block_data = block_data[0:(latest_fork - 1)]
    block_data = block_data.append(replacement_data)
    for b in range(latest_fork, max_block):
        cube = remove_block(cube, b + 1)
    max_block = get_max_block(cube)
    return(block_data, cube, max_block, lowest_fork)

# Function that creates a queu of objects to run parallel process on.
# Create the queue starting with the lowest fork and then moving upwards until
# we've reached the number of CPU cores.
# =============================================================================
# This function was depricated for a more robust function that could create
# a queue on any number of cores. This function couldn't handle anything higher
# than the sum of permutations above the second level.
# def create_queue(cube, block_data, n_cores):
#     block_data = sf.get_lengths(block_data)
#     lowest_fork = int(min(block_data[block_data['len'] > 1].block))
#     fork_size = block_data[block_data.block == lowest_fork].len[0]
#     block_data = block_data.drop(['len'], axis = 1)
#     # Create subset that contains all lowest non-forked rows
#     # block_data_sub = block_data[0:(lowest_fork - 1)]
#     sub_queue = []
#     for s in range(fork_size):
#         block_data_queue = split_fork(block_data, lowest_fork, s)
#         sub_queue.append(block_data_queue)
#     
#     queue = []
#     for q in range(len(sub_queue)):
#         sub_queue[q] = sf.continue_path(cube, sub_queue[q])[0]
#         sub_queue[q] = sf.get_lengths(sub_queue[q])
#         max_block = int(max(sub_queue[q].block[0]))        
#         new_direction = sub_queue[q][sub_queue[q].block == max_block - 1].directions[0][0]
#         cube_dummy = copy.deepcopy(cube)
#         cube_dummy = sf.add_block(cube_dummy, block = max_block, \
#                                 direction = new_direction)
#         lowest_fork = int(min(sub_queue[q][sub_queue[q]['len'] > 1].block))
#         fork_size = sub_queue[q][sub_queue[q].block == lowest_fork].len[0]
#         sub_queue[q] = sub_queue[q].drop(['len'], axis = 1)
#         for s in range(fork_size):
#             block_data_queue = split_fork(sub_queue[q], lowest_fork, s)
#             queue.append(block_data_queue)
#     
#     if len(queue) > n_cores:
#         queue = queue[0:n_cores]
#     
#     return(queue)
# =============================================================================
    