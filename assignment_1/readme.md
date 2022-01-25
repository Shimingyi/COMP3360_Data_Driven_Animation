# Assignment 1 - Basic Character Animation

In this assignment, you will learn the basic data structure and animation creation pipeline and are required to create an animation clip with provided Infrastructure. Also, you need to understand the mathematics in FK and IK for reading the motion capture files and playing with them.

All software-related codes will be provided by @[Mingyi Shi](https://rubbly.cn/) and @[He Zhang](https://cghezhang.github.io/), only the core codes need to be filled in.

![](https://user-images.githubusercontent.com/7709951/150998067-5652b8aa-54fc-43e4-8eb6-2020ebe2067c.png)

Due to: **13th. Feb**

Submission: A compress file with [video_file, bvh_visualizer.py, CCDIK.cs, uid_name_1.pdf] named by uid_name_assignment1.zip 



### Quiz 1 - A rendered video with character animation (15%)

1. Download Blender
2. Import the provided FBX file and check the mesh (*./data/angry_girl.fbx*)
3. Open provides blender file to check the content in *quiz1/\** folder 

- - **skeleton_in_mesh.blend**   
  - **rig_body_blocks.blend** and **rig_body_mesh.blend**
  - **animation_keyframe.blend** and **animation_bvhs.blend**

4. To-Do: Create a keyframe animation with **exersise_k_frame.blend** and render it to a video (without motion quality requirement).



### BVH visualizer  (45%)

1. Some blender examples have been prepared in ***example.py***, you can find how to add different mesh into Blender by python scripts and how to make animation with keyframing way
2. The goal of this part is to understand basic motion data structure and know the mathematical operation between different data types, like rotation and position.
3. For convenience, we put all the helper functions on the same file, including the definition of Quaternion class (L6-L425) and BVH file parser(L428-L555), you don't need to write anything in this part. 
4. Given a BVH file, there are different ways to visualize it. One is based on rotation, which has been implemented by TA, you can check it from L561 to L623. We firstly put all the joints and bones on the scene refer to the offset, and then set the keyframe with rotation to animate the skeleton. The pipeline will be similar when we use positions for animation.
5. Before you run it in Blender, you should replace the file path in L661. The script will display the rotation-based visualization by default, so you can compare it with your implements.
6. To-Do: Open ***bvh_visualizer.py***, implement functions **build_joint_cubes** and **set_cube_animation**. Two main tasks are required here, in *build_joint_cubes*, you should convert joint offsets to global joint positions, and in set_cube_animation, you will use rotations to update global joint positions in each frame and insert it as a keyframe. 
7. The code is expected to be runnable.



### IK solver (30%)





### Report (10%)

1. No word limitation and requirement, just record your ideas. 

2. It's hard for TA to run all the scripts (maybe lots of bugs), so screenshots are expected to appear in this report to show your final results.

3. If you cannot implement the functions, please explain more here to tell what method you tried and how does it not work. 

4. Try to answer these open questions:

5. 1. Why rotation is more common? What's the problem if we only use joint position to record motion?
   2. How about using IK in real-time gaming? Do you have any idea how to improve it?