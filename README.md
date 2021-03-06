# COMP3360 Data-Driven Animation
The code repository of HKU course COMP3360

Instructor: [Prof. Taku Komura](https://www.cs.hku.hk/index.php/people/academic-staff/taku)

TAs: @[Mingyi Shi](https://rubbly.cn) , @[He Zhang](https://cghezhang.github.io) and @[Floyd Chitalu](https://github.com/chitalu)

![cover](https://user-images.githubusercontent.com/7709951/150430601-470046fb-7370-48cb-8ee5-af8765b6f064.png)

## Assignment 1 - Basic Character Animation

In this assignment, you will learn the basic data structure and animation creation pipeline and are required to create an animation clip with provided Infrastructure. Also, you need to understand the mathematics in FK and IK to read the motion capture files and play with them.

All software-related codes will be provided by @[Mingyi Shi](https://rubbly.cn) and @[He Zhang](https://cghezhang.github.io). Only the core codes need to be filled in.

All the materials have been uploaded: [subfolder](./assignment_1)

#### Tutorial Slides

1. Data Structure in Character Animation [[slides](./tutorial1_data_structure.pdf)]
2. Scripting with BVH file and IK solver [[slides](./tutorial2_scripting.pdf)]

#### Assessments (due to 11:59 am, 18th. Feb)

* A rendered video with character animation (Quiz1, 15%)
* BVH file visualizer in Blender/Unity (45%)
* An IK solver in Unity (30%)
* Report (10%)

## Assignment 2 - Data-Driven Character Animation

This assignment will teach you how to observe motion data and process it with different tools like the mathematics interpolation algorithm and AutoEncoder model. Given a motion clip shaped with (T, J, R), understanding the difference between temporal and spatial dimensions will be the key to processing it. You will practice it by following tasks.

All the materials have been uploaded: [subfolder](./assignment_2)

#### Tutorial Slides (2 Tutorials for each Assignment)

1. Introductory Motion Data Processing [[slides](../tutorial3_motion_processing.pdf)]
2. Data-Driven Motion Processing [[slides](../tutorial4_data_driven_motion_processing.pdf)]

#### Assessments (due to 11:59 am, 18th. Mar)

- Mathematics interpolation interface for keyframing animation and motion concatenation (40%)

- Data-Driven model for motion denoising and interpolation (60%)

- Report

## Physics-based animation tutorials

IMPORTANT NOTES ABOUT TUTORIAL 1 AND TUTORIAL 2: 
* For those of you having issues on Windows due to CMake and GLFW, please use the project template that is provided [here](https://github.com/chitalu/glfw-project). Follow the instructions in the corresponding readme file to build the code, which creates a simple window with a green background.
* Once you have setup the template, you can then copy the `main.cpp` and `.obj` files that were provided for the tutorial (1 or 2) you are working on. Do not copy the remaining files that are provided in the original tutorial 1 and 2 templates. (also dont copy the extern/ directory).
* Follow the usual instructions to generate the Visual Studio files (i.e. `mkdir build`, `cd build` and then `cmake ..`). The generated files will be in the created build directory.
* If you get build errors, it is likely that you need to rename the following two variables names `near` and `far` to `near_plane` and `far_plane` (there is a clash of variable names with one of the Windows system header files). 

### Tutorial 1 - Rigid body dynamics

This tutorial will teach you how to implement a basic rigid body dynamics solver in C++. 

All the materials to help you get started have been uploaded to the folder [pba_tutorial1_rbd](./pba_tutorial1_rbd). Start by reading the README.md file in that directory.

The tutorial slides can be found [[here](./pba_tutorial1_rbd.pdf)].

### Tutorial 2 - Cloth dynamics

This tutorial will teach you how to implement a basic cloth simulator in C++. 

All the materials to help you get started have been uploaded to the folder [pba_tutorial2_cloth](./pba_tutorial2_cloth). Start by reading the README.md file in that directory.

The tutorial slides can be found [[here](./pba_tutorial2_cloth.pdf)]. 

### Tutorial 3 - FEM deformation dynamics

This tutorial will teach you how to implement volumetric deformation using FEM in C++. 

All the materials to help you get started have been uploaded to the folder [pba_tutorial3_fem](./pba_tutorial3_fem). 

The tutorial slides can be found [here](./pba_tutorial3_fem.pdf) 

Here are two examples of the expected output:

![output1](./pba_tutorial3_fem/expected-output1.png)

![output2](./pba_tutorial3_fem/expected-output2.png)

### :warning: Assignment 3 : Hyperelastic deformation

In this assignment, you will work with the C++ programming language to learn about implementing
different hyperelastic material models and numerical integration schemes that you encountered during the lectures.

The final assignment should be submitted as a .zip file which contains only the source files of your
project. Do not include pre-compiled binaries (!) unless deemed absolutely necessary. For non-coding tasks,
please submit additional files that contain your answers.

This assignment is in essence a direct extension of Tutorial 3 (FEM deformation dynamics), all software-related codes will uploaded to the folder [pba_tutorial3_fem](./pba_tutorial3_fem).

#### Deadline

The deadline for this assignment is 22 April 2022

#### Instructions

Refer to the document [here](./assignment_3/instructions.pdf) for detailed instructions about this assignment. 

Pls, contact myshi@cs.hku.hk or chitalu@hku.hk if there is any question.
