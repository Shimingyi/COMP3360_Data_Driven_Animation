# COMP3360 Data-Driven Computer Animation

Welcome to the COMP3360 in year 2023!

Here is the code repository of HKU course COMP3360. Any commit and issue will be welcome.

Instructor: [Prof. Taku Komura](https://www.cs.hku.hk/index.php/people/academic-staff/taku)

TAs: @[Mingyi Shi](https://rubbly.cn)  @[Huancheng Lin](https://github.com/LamWS)

![cover](https://user-images.githubusercontent.com/7709951/212983788-cf6feaed-c81c-4b99-8638-d7cf2a8f9328.jpg)

## Instruction

* Always keep the latest version of this repo
  ```
  git clone https://github.com/Shimingyi/COMP3360_Data_Driven_Animation.git -b 2023
  ```
* Don't hesitate to seek helps with issue workspace

## Assignment 1 - Basic Character Animation

In this assignment, you will learn the basic data structure and animation creation pipeline and are required to create an animation clip with provided infrastructure. Also, you need to understand the mathematics in FK and IK to read the motion capture files and play with them.

Details: [[subfolder](./assignment_1)]

#### Tutorial Slides

1. Basic Linear Algebra in Graphics [[slide](./COMP3360_ANI_T1.pdf)]
2. Forward and Inverse Kinematics [[slide](./COMP3360_ANI_T2.pdf)]

#### Assessments

- A rendered video with character animation (Task 1, 40%)
- Core code implementation of Forward Kinematics (Task 2, 25%)
- Core code implementation of Inverse Kinematics - CCD IK (Task 3, 25%)
- Report (10%)

## Assignment 2 - Animation Processing and Scripting

This assignment will provide a practical introduction to working with animation data through various algorithms such as interpolation and concatenation. Additionally, you will learn to consider various variables from motion data to enhance the performance of the motion matching method.

Detials: [[subfolder](./assignment_2)]

#### Tutorial Slides

1. Basic motion processing [[slides](./COMP3360_ANI_T3.pdf)]
2. Interactive Animation System [[slides](./COMP3360_ANI_T4.pdf)]

#### Assessments

* part1_key_framing (30%)
  * Linear interpolation (10%); Slerp Interpolation (15%)
  * Report the different performance by giving different numbers (5%)
* part2_concatenation (35%)
  * Define the search window (10%) + Calculate the sim_matrix (10%);
  * Find the real_i and real_j (10%);
  * The shifting on the root joint position (5)
* part3_motion_matching (25%)
  * A working system (10%)
  * Variable terms (22% - your_variable_num)
* Report (8%) + 2 videos (2%)
  * Including necessary experiment results by *different parameters* (4%) and your *thinking*(4%) for how to produce high quality motions.

## Assignment 3 - TBA

Pls, contact myshi@cs.hku.hk or lamws@connect.hku.hk if there is any question.
