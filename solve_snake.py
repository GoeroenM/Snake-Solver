import numpy as np
import pandas as pd
import os
import sys

# Change path to script location
# os.chdir(os.path.realpath(sys.argv[0])) # This used to work but not anymore
os.chdir("C:\\Users\\goero\\OneDrive\\Documenten\\Snake-Solver\\")

# Load definition of snake
snake = pd.read_csv("define_snake.txt", sep = "\t", header = 0)

# Define necessary functions
def add_block(cube, block, from_location, direction):
    new_position = np.array(from_location) + np.array(direction)
    # Check if the new position is within the bounds of the cube
    if any(x < 0 for x in new_position) or any(x > 3 for x in new_position):
        raise ValueError("Block out of bounds.")
    # Check if the new position is already taken
    elif cube[new_position[0]][new_position[1]][new_position[2]] != 0:
        raise ValueError("Position already taken.")
    else:
        cube[new_position[0]][new_position[1]][new_position[2]] = block
    return(cube)

def get_block_number(cube, position):
    position = np.array(position)
    return cube[position[0]][position[1]][position[2]]

def get_max_block(cube):
    return(int(max(np.unique(cube))))

def get_block_location(cube, block_number):
    return(np.array(np.where(cube == block_number)).flatten())

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
        # For both axes, need to create 2 directions, the previous one is 0
        for axis in other_axes:
            for x in range(2):
                possible_directions[dummy][last_axis] = 0
                possible_directions[dummy][axis] = (-1)**x
                dummy = dummy + 1
        
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
        possible_directions = get_last_direction(cube)
    return(possible_directions)

# Initialize cube with known positions
cube = np.zeros((4,4,4))
cube[3][1][0] = int(1)
cube[3][0][0] = int(2)
cube[2][0][0] = int(3)
cube[1][0][0] = int(4)
cube[0][0][0] = int(5)

max_block = get_max_block(cube)
last_direction = get_last_direction(cube)

# Initialize block metadata
columns = ["Block", "Elbow", "location", "directions"]
block_data = pd.DataFrame(columns = columns)

for block in np.unique(cube)[np.unique(cube) != 0]:
    block_number = block
    block_location = get_block_location(cube, block_number)
    elbow = (int(snake[snake.Block == max_block]['Elbow']) == 1)
    if block_number < max_block:
        direction = get_direction(cube, block_number, (block_number + 1))
    else:
        # Set the elbow variable as boolean, rather than 0/1
        elbow = (int(snake[snake.Block == max_block]['Elbow']) == 1)
        direction = get_possible_directions(cube, block_number, elbow)
    block_data = block_data.append(pd.DataFrame({"Block":block_number,
                                    "Elbow":elbow, 
                                    "location":[block_location],
                                    "directions":[direction]}))
    
# blocks_to_go = np.setdiff1d(snake["Block"], np.unique(cube))
blocks_to_go = snake[max_block:]

