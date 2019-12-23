# Snake-Solver
I wrote this code to help me solve a puzzle I bought at a fair. The puzzle was a cube, which was comprised of smaller blocks in a 4x4 grid. The smaller blocks are all connected by a string. Some blocks act as an 'elbow' around which you can turn the other blocks, while others are not. The reason I call it a snake solver is because the puzzle was called a 'snake' by the guy that sold it to me.

After I had unfolded the cube a first time and had tried solving it without much success, I decided to try and solve it using the 'solution' that was given alongside the cube at the moment of purchase. I quickly found out that this solution was absolutely unreadable, which meant I couldn't solve the cube.

I was able to deduce the final positions of a handful of blocks based on this solution manual and used these as a seed to write a program that would solve the cube for me.

The principle is as follows: I define the string of connected blocks by their number in the chain and whether or not the block itself is an 'elbow'. I define an empty 4x4 cube and fill it with the positions I know. The metadata of the cube is saved as a dataframe which contains the block number, its location, elbow and all possible directions away from that block. A direction is always from the block itself to the next one.
The code will then place the next block based on the first possible direction from the last one and keeps doing this until it finds a dead end. At this point, it goes back down one level and removes the first of all possible directions in the lower level. If the lower level has no possible directions left, the code works its way down to the latest fork, which is the last block from which more than one direction could have been taken. Here, it removes the first direction (as it was tried already) and tries the next one.

This keeps going until the cube is solved.

Note: since the time required to solve the cube increases exponentially for each extra block you need to place, you'll need to know the starting positions of at least ~10 blocks, or the code will take a ridiculous amount of time to run. At 11 blocks, it took me

define_snake.txt and define_snake_reverse.txt contain the definitions of the 'snake': each block and whether or not it is an elbow.

initiate_cube.txt and initiate_cube_reverse.txt contain the initial positions of blocks used as a seed to solve the cube. It could be done without any known positions, but I don't think there's enough time left until Earth is consumed by the expanding Sun to calculate it that way.

snake_functions.py contain a bunch of predefined functions used to deal with the cube (add/remove blocks, check directions, etc).

solve_snake.py solves the cube based on the input in a linear fashion. Just one process that tries to solve it.

solve_snake_parallel.py is a work-in-progress where I try to solve the cube using parallel processes. It doesn't currently work yet.
