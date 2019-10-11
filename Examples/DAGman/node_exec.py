#!/usr/bin/env python
# coding: utf-8


import numpy as np
import argparse



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=str, help="Position in Grid")
    args = parser.parse_args()

    print(args.pos)
    
    # Setup local compute grid
    n_gridpoints = 10 - 1
    local_grid_points = n_gridpoints//3 +1
    local_grid = np.zeros((local_grid_points, local_grid_points))

    # Read interfacing values
    x_pos = int(args.pos[0])
    y_pos = int(args.pos[1])

    # load bordering values
    xm1y = np.load(str(x_pos-1)+str(y_pos)+"_E.npy")
    xm1ym1 = np.load(str(x_pos-1)+str(y_pos-1)+"_NE.npy")
    xym1 = np.load(str(x_pos)+str(y_pos-1)+"_N.npy")

    # copy bordering values into local_grid
    local_grid[1:, 0] = xm1y
    local_grid[0, 0] = xm1ym1
    local_grid[0, 1:] = xym1

    # load transitions matrix
    local_grid_transitions = np.load("grid_transitions_"+args.pos+".npy")

    for x_i in range(1, local_grid_points):
        for y_i in range(1, local_grid_points):
            local_grid[x_i, y_i] = local_grid_transitions[x_i-1, y_i-1] @ np.array([local_grid[x_i-1, y_i],
                                                                                local_grid[x_i-1, y_i-1],
                                                                                local_grid[x_i, y_i-1]])

    np.save("res_"+args.pos, local_grid[1:, 1:])
    np.save(args.pos+"_N", local_grid[local_grid_points-1, 1:])
    np.save(args.pos+"_E", local_grid[1:, local_grid_points-1])
    np.save(args.pos+"_NE", local_grid[local_grid_points-1, local_grid_points-1])

