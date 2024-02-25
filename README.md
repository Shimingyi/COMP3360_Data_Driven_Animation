# COMP3360 Data-Driven Computer Animation

Welcome to the COMP3360 in year 2023-2024!

Here is the code repository of HKU course COMP3360. Any commit and issue will be welcome.

Instructor: [Prof. Taku Komura](https://www.cs.hku.hk/index.php/people/academic-staff/taku)

TAs: @[Mingyi Shi](https://rubbly.cn)  @[Zhouyingcheng Liao](https://zycliao.com/)

![cover](https://github.com/Shimingyi/COMP3360_Data_Driven_Animation/assets/7709951/87c572b8-cd20-4c97-8922-34fb84ba1660)

## Instruction

* Get the latest version of this repo
``` shell
git clone https://github.com/Shimingyi/COMP3360_Data_Driven_Animation.git -b 2024
```
* Don't hesitate to seek helps with issue workspace

## Assignment 1 - Basic Character Animation

In this assignment, you will become familiar with fundamental data structures and the animation creation workflow. Your task is to craft an animation clip using the provided infrastructure.
Also, you need to understand the mathematical concepts behind Forward Kinematics (FK) and Inverse Kinematics (IK), and then to interpret the motion capture files and interact with them.

Details: [[subfolder](./assignment_1)]

#### Tutorial Slides

1. Basic Linear Algebra in Graphics [[slide](./COMP3360_ANI_T1.pdf)]
2. Forward and Inverse Kinematics [[slide](./COMP3360_ANI_T2.pdf)]

#### Assessments

- A rendered video with character animation (Task 1, 40%)
- Core code implementation of Forward Kinematics (Task 2, 25%)
- Core code implementation of Inverse Kinematics - CCD IK (Task 3, 25%)
- Report (10%)

#### Useful tutorials of Blender
 - [Basic operations of Blender](https://www.youtube.com/watch?v=B0J27sf9N1Y)
 - [Character rigging](https://www.youtube.com/watch?v=9dZjcFW3BRY)
 - [Keyframing animation](https://youtu.be/yjjLD3h3yRc?si=_-X3Nb3PRaNWeq6h) 

## Assignment 2 - Animation Processing and Scripting

This assignment will provide a practical introduction to working with animation data through various algorithms such as interpolation and concatenation. Additionally, you will learn to consider various variables from motion data to enhance the performance of the motion matching method.

Detials: [[subfolder](./assignment_2)]

#### Tutorial Slides

1. Basic motion processing [[slides](./COMP3360_ANI_T3.pdf)]

#### Assessments

* part1_key_framing (30%)
  * Linear interpolation (10%); Slerp Interpolation (15%)
  * Report the different performance by giving different numbers (5%)
* part2_concatenation (35%)
  * Define the search window (10%) + Calculate the sim_matrix (10%);
  * Find the real_i and real_j (10%);
  * The shifting on the root joint position (5)
* part3_motion_matching (25%)
  * TBA
* Report (8%) + 2 videos (2%)
  * Including necessary experiment results by *different parameters* (4%) and your *thinking*(4%) for how to produce high quality motions.

Pls, contact zycliao@cs.hku.hk or myshi@cs.hku.hk if there is any question.
