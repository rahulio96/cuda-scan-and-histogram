# CUDA Parallel Patterns

## Description
This project was part of an assignment for a Parallel Programming with GPUs course I took during Spring 2025.

The goal was to develop a GPU kernel that showcases two parallel patterns:
1. List Scan (Parallelized Prefix Sum)
2. Histogram

Used Nvidia's CUDA progrmaming language to write the kernels.

## How to Run
1. Download libwb and add it under the `source` directory
2. Create a new `build` directory under the root directory
3. Use the CMake GUI and set the source and build path to the `source` and `build` directories respectively and generate build files
4. Build the solution using Visual Studio and run commands inside `A3_batch.txt` to test the code on data from the `datasets` directory