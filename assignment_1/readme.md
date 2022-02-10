# Assignment 1 - Basic Character Animation



## Hightlights (11th, Feb)

- The bvh visualizer should be implemented by **inserting locations** as keyframes rather than rotation, hence a **forward kinamatic function** will be expected to be here to convert the joint rotation to position. 
- Check the [Tips](https://github.com/Shimingyi/COMP3360_Data_Driven_Animation/tree/main/assignment_1#tips ) if you meet runtime problems or stuggled in debugging. 



## Introduction 

In this assignment, you will learn the basic data structure and animation creation pipeline and are required to create an animation clip with provided Infrastructure. Also, you need to understand the mathematics in FK and IK for reading the motion capture files and playing with them.

All software-related codes will be provided by @[Mingyi Shi](https://rubbly.cn/) and @[He Zhang](https://cghezhang.github.io/), only the core codes need to be filled in.

![](https://user-images.githubusercontent.com/7709951/150998067-5652b8aa-54fc-43e4-8eb6-2020ebe2067c.png)

#### Tutorial Slides (2 Tutorials for each Assignment)

1. Data Stucture in Charater Animation [[slides](../tutorial1_data_structure.pdf)]
2. Scripting with BVH file and IK solver [[slides](../tutorial2_scripting.pdf)]

#### Submission

Due to: **11:59am 18th, Feb**

File format: A compress file with [video_file, bvh_visualizer.py, CCDIK.cs, uid_name_1.pdf] named by uid_name_assignment1.zip 



## Quiz 1 - A rendered video with character animation (15%)

1. Download Blender
2. Import the provided FBX file and check the mesh (*./data/angry_girl.fbx*)
3. Open provides blender file to check the content in *quiz1/\** folder 

- - **skeleton_in_mesh.blend**   
  - **rig_body_blocks.blend** and **rig_body_mesh.blend**
  - **animation_keyframe.blend** and **animation_bvhs.blend**

4. To-Do: Create a keyframe animation with **exersise_k_frame.blend** and render it to a video (without motion quality requirement).



## BVH visualizer  (45%)

- Some blender examples have been prepared in ***example.py***, you can find how to add different mesh into Blender by python scripts and how to make animation with keyframing way

- The goal of this part is to understand basic motion data structure and know the mathematical operation between different data types, like rotation and position.

- For convenience, we put all the helper functions on the same file, including the definition of Quaternion class (L6-L425) and BVH file parser(L428-L555), you don't need to write anything in this part. 

- Given a BVH file, there are different ways to visualize it. One is based on rotation, which has been implemented by TA, you can check it from L561 to L623. We firstly put all the joints and bones on the scene refer to the offset, and then set the keyframe with rotation to animate the skeleton. 
- Before you run it in Blender, you should replace the file path in L661. The script will display the rotation-based visualization by default, so you can compare it with your implements.

Requirements:

1. To-Do: Open ***bvh_visualizer.py***, implement functions **build_joint_cubes** and **set_cube_animation**. Two main tasks are required here, in *build_joint_cubes*, you should convert joint offsets to global joint positions, and in set_cube_animation, you will **convert the rotations to position** firstly, and then insert **joint position** as a keyframe (not rotation). 

   ```python
   def build_joint_cubes(offsets, parents, names, parent_obj, all_obj):
       
       return 
     
   def set_cube_animation(joints, rotations, root_position, parents, frametime):
       global_position = np.zeros((rotations.shape[0], rotations.shape[1], 3)) 
       ## Your implement here: Convert the rotations to positions
   
       
       for frame_idx in range(rotations.shape[0]):
           for joint_idx in range(rotations.shape[1]):
               joints[joint_idx].location = global_position[joint_idx]
               joints[joint_idx].keyframe_insert(data_path='location', frame=frame_idx) # You should insert keyframe with location
       return
   ```

2. The code is expected to be runnable.



## IK solver (30%)

1. Download Unity and activate the free personal license.
2. Import the course project refer to the introductions in the tutorial slides.
3. Try the provided IK script, and understand the meaning of IKTip, IKRoot, and IKTarget.
4. For convenience, all the needed files and functions have been prepared in this project, so all your implements can be done in CCDIK.cs.
5. To-Do: Open ***CCDIK.cs***, implement function **SolveByCCD**. Two main tasks are required here, at first you need to solve Tip Position based on CCD solution, and then heuristic IK weight computation strategy will be used to improve the quality. 
6. The code is expected to be runnable.



## Report (10%)

1. No word limitation and requirement, just record your ideas. 

2. It's hard for TA to run all the scripts (maybe lots of bugs), so screenshots are expected to appear in this report to show your final results.

3. If you cannot implement any feature, please explain what method you tried and how does it not work. 

4. Try to answer these open questions:

    1. Why rotation is more common? What's the problem if we only use joint position to record motion?
    
    2. How about using IK in real-time gaming? Do you have any idea how to improve it?
    
       

## Tips

1. How to print logs with Blender Python?

   A: Configure the blender exe/binary to system path, then open blender with terminal, all the logs will be displayed as outpout. Or you can install [add-on](https://b3d.interplanety.org/en/blender-add-on-print-to-python-console/) to support it.

2. Try to debug your function in a blender-free environment

   A: Download a code editor, like VS code, or an IDE to debug the no-blender-dependent file, like what we provided as [bvh_visualizer_debug.py](https://github.com/Shimingyi/COMP3360_Data_Driven_Animation/blob/main/assignment_1/bvh_visualizer_debug.py) . You can make breakpoint and check the variables one by one[[official tutorial](https://code.visualstudio.com/docs/python/python-tutorial)]. In this file, I prepare a toy data with only three joints, with initial location (1, 0, 0), (2, 0, 0) and (3, 0, 0). Then I apply eular rotation (0, 0, 45) on joint 1 and joint 2, finally I got the poseture like below image. Your implementation in the debug file should reach same result, then you can consider copy the correct function into blender scripts. 

<img width="1218" alt="example" src="https://user-images.githubusercontent.com/7709951/153510650-062da828-ad1e-41a3-9ffc-1fde6cc6144a.png"> 

3. ImportError: cannot import name '_ccallback_c' from 'scipy._lib' (C:\Program Files\Blender Foundation\Blender 3.0\3.0\python\lib\site-packages\scipy\_lib\__init__.py) 

   ```bash
   Solution: Run Command Prompt as Administrator
   cd C:\Program Files\Path\To\Blender\python\bin
   .\python -m pip install --no-deps -t "C:\Program Files\Blender Foundation\Blender 3.0\3.0\python\lib\site-packages" scipy
   ```
