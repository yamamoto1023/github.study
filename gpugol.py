#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import curses
from curses import wrapper

from pycuda.compiler import SourceModule

SIZE_X = 100
SIZE_Y = 100
BLOCKSIZE = 32

row2str = lambda row: ''.join(['O' if c != 0 else '-' for c in row])

def print_world(stdscr, gen, world):
    '''
    盤面をターミナルに出力する
    '''
    stdscr.clear()
    stdscr.nodelay(True)
    scr_height, scr_width = stdscr.getmaxyx()
    height, width = world.shape
    height = min(height, scr_height)
    width = min(width, scr_width - 1)
    for y in range(height):
        row = world[y][:width]
        stdscr.addstr(y, 0, row2str(row))
    stdscr.refresh()

    
mod = SourceModule("""
__global__ void calc_next_cell_state_gpu(const int* __restrict__ world, int* __restrict__ next_world, const int mat_size_x, const int mat_size_y) {
    int mat_x = threadIdx.x + blockIdx.x * blockDim.x;
    int mat_y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = mat_y * mat_size_x + mat_x;
    int cell, next_cell;
    int num = 0;

    if(mat_x >= mat_size_x){
        return;
    }
    if(mat_y >= mat_size_y){
        return;
    }

    cell = world[index];

    num += world[((mat_y - 1) % mat_size_y) * mat_size_x + ((mat_x-1) % mat_size_x)];
    num += world[((mat_y - 1) % mat_size_y) * mat_size_x + ((mat_x) % mat_size_x)];
    num += world[((mat_y - 1) % mat_size_y) * mat_size_x + ((mat_x+1) % mat_size_x)];
    num += world[(mat_y % mat_size_y) * mat_size_x + ((mat_x-1) % mat_size_x)];
    num += world[(mat_y % mat_size_y) * mat_size_x + ((mat_x+1) % mat_size_x)];
    num += world[((mat_y + 1) % mat_size_y) * mat_size_x + ((mat_x-1) % mat_size_x)];
    num += world[((mat_y + 1) % mat_size_y) * mat_size_x + ((mat_x) % mat_size_x)];
    num += world[((mat_y + 1) % mat_size_y) * mat_size_x + ((mat_x+1) % mat_size_x)];

    if (cell == 0 && num == 3){
        next_cell = 1;
    }else if(cell != 0 && (num >= 2 && num <= 3)){
        next_cell = 1;
    }else{
        next_cell = 0;
    }
    next_world[index] = next_cell;
}
""")


def calc_next_world_gpu(world, next_world):
    """
    現行世代の盤面を元に次世代の盤面を計算する
    """
    block = (BLOCKSIZE, BLOCKSIZE, 1)
    grid = ((SIZE_X + block[0] - 1) // block[0], (SIZE_Y + block[1] -1) //block[1])
    height, width = world.shape
    calc_next_cell_state_gpu = mod.get_function("calc_next_cell_state_gpu")

    calc_next_cell_state_gpu(cuda.In(world), cuda.Out(next_world), numpy.int32(SIZE_X), numpy.int32(SIZE_Y), block = block, grid = grid)


def gol(stdscr, width, height):
    world = numpy.random.randint(2, size = (width, height), dtype=numpy.int32)

    gen = 0
    while True:
        print_world(stdscr, gen, world)
        next_world = numpy.empty((height, width), dtype=numpy.int32)
        calc_next_world_gpu(world, next_world)

        world = next_world.copy()
        gen += 1


def main(stdscr):
    gol(stdscr, SIZE_X, SIZE_Y)

if __name__ == '__main__':
    curses.wrapper(main)
