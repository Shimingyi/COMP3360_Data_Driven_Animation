# Getting started

## Dependencies

To compile the code, you will need to the following:

* GLFW, which can be obtained from here [http://www.glfw.org/download.html]. It 
is recommend that you install as a package using your favourite package manager.
* OpenGL: In all three major desktop platforms (Linux, macOS, and Windows), 
OpenGL more or less comes with the system.
* CMake, which is a tool to generate project/solution files of your choice 
(e.g. Visual Studio, Code::Blocks, Eclipse). You can download CMake from 
here [http://www.cmake.org/cmake/resources/software.html], or as a package using 
your favourite package manager.

## Compilation

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

Implement a cloth simulator. You are only expected to implement the cloth _dynamics_ (with some constraints/
fixed vertices and gravity). The sections of code that you are required to edit are marked TODO

All the necessary files have been provided. You will need to implement the logic 
of two main functions: 
* `void init_physical_object(...)`, and 
* `void update_physics(...)`

###  Extras

Attempt to add some extra to your simulator:
* different integration scheme 
* collision detection 
* basic interation e.g. with mouse or keyboard.
* Implicit integration (Newton Ralphson solver)
    - You are allowed to add third-party libraries to your project e.g. "Eigen",
    which is a high-level C++ library of template headers for linear algebra, 
    matrix and vector operations, geometrical transformations, numerical solvers 
    and related algorithms. 

