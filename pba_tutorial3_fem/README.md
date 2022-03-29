# Getting started

## Dependencies

* CMake, which is a tool to generate project/solution files of your choice 
(e.g. Visual Studio, Code::Blocks, Eclipse). You can download CMake from 
here [http://www.cmake.org/cmake/resources/software.html].

## Code setup and compilation

### If using a terminal

* `mkdir build`
* `cd build`
* `cmake ..` (you may also use the CMake GUI for this step)
* `make` if you're on linux/MacOS. Otherwise, simply open the generated .sln 
with Visual Studio and compile if you are on Windows.

### If using CMake GUI

* Create a sub-directory called `build`
* Configure you project using the CMake GUI as described (here)[https://cmake.org/runningcmake/]

## Your task

Implement an FEM simulator _using the Linear Elasticity model_. The sections of code that you are required to implement are marked TODO.

All the necessary files have been provided. You will need to implement the logic 
of two main functions: 
* `void init_physical_object(...)`, and 
* `void update_physics(...)`
* You will also need to implement/define the necesary structs at the top of `main.cpp` 

## Scene setup

A cylinder that aligned along the x-axis, and will be pulled (to cause tension) and pushed (to cause compression).

To determine whether the cylinder should be pulled use `apply_pull_force = true`, which is declared at the top of `main.cpp` (line 54). Alternatively, you may set `apply_pull_force = false` for compression. 

