# Summary for Assignment 1

Thanks for all of your hard-working on assgnment1. This course is an advanced course that requires more programming and a computer graphics background, and I am so happy to see your submissions, though it's a challenge for some junior students. Some submissions cannot satisfy all requirements, but you can still get marks once you share your basic idea in the report; it's essential in this course because not all students have strong programming skills.

All of you did an awesome animation on task 1, and I selected some examples into the gallery; in the beginning, some students even don't know how to apply a rotation on a vector, but finally, half of you can reach 90 marks, which makes a big success; task 3 requires more experimental analysis, and all students did a good job.  

I will give a detailed explanation for the BVH visualizer part. 



## Blender Animation 

Hope you enjoy these awesome animations created by students

![video3](https://user-images.githubusercontent.com/7709951/158897332-666eaca7-5011-4b21-babb-44c3a127a354.gif)

![video2](https://user-images.githubusercontent.com/7709951/158897304-7759b671-0a62-4c64-934c-d6be46fdbca1.gif)

![video1](https://user-images.githubusercontent.com/7709951/158897360-238bbdd9-f058-4f78-9c06-1388e806c635.gif)



## BVH Visualizer

The BVH visualizer may be the main challenge in Assignment 1, and I am sorry for the little help. Almost students can get an idea of how to do the calculation, but some bugs make the final animation strange, which is also hard to locate because of the Blender environment. 

#### Basic knowledge

I will introduce this basic knowledge initially; some students meet a big gap because of the lack of this part. 

* The entry point of the python script

  A good explanation has been introduced in this [blog](https://towardsdatascience.com/entry-point-of-a-python-application-c001ff15a355). About our script, the entrance is located in the last lines starting with [L670](https://github.com/Shimingyi/COMP3360_Data_Driven_Animation/blob/main/assignment_1/bvh_visualizer.py#L670). 

  Here, we first load the data from the motion file and then send these data into the <u>build_reset_skeleton</u> and <u>set_animation</u> function. In your code, you should replace these two functions with <u>build_joint_cubes</u> and <u>set_cube_animation</u>, which you have implemented.

  ```python
  rotations, positions, offsets, parents, names, frametime = load(filename='./assignment1/data/motion_files/motion_basket.bvh')
  
  all_obj = []
  build_reset_skeleton(offsets, parents, names, 0, None, all_obj)
  .....
  set_animation(all_joints, positions, rotations.qs, frametime)
  bpy.ops.object.select_all(action='DESELECT')
  ```

* The representation for data

  There can be different representations for rotation data, like [Quaternion](https://en.wikipedia.org/wiki/Quaternion) with four numbers(w, x, y, z) or [Euler](https://en.wikipedia.org/wiki/Euler_angles) with three numbers (x, y, z). 

  For position data, there are three numbers to represent one position - x, y, and z. 

  We can use vector to represent the limb, which also includes three number, d_x, d_y, and d_z, which is calculated by:

  d_x, d_y, d_z = joint 1 - joint 2 = (x1, y1, z1) - (x2, y2, z2)

* Applying rotation on a vector

  There are lots of ways to present the rotation process. 

  ```python
  # 2d rotation, numpy style, source from https://www.atqed.com/numpy-rotate-vector 
  import numpy as np
  theta = np.deg2rad(30)
  transforms = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
  vector = np.array([1, 0])
  vector_rotated = np.dot(transforms, vector)
  
  # 3d rotation, scipy style, source from https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.apply.html
  from scipy.spatial.transform import Rotation as R
  vector = np.array([1, 0, 0])
  r = R.from_rotvec([0, 0, np.pi/2])
  r.apply(vector)
  ```

#### Pipeline

From the tutorial, we will know there are two main parts in a BVH file. The first OFFSET module stores the hierarchical joint structure, and the second CHANNEL module holds the position and rotation values. These two parts will be used in our tasks.

For more details, in this task, you are required to put cubes on the joint positions only based on its positional information, so there are two main steps:

1. put cubes on the scene, the number of cubes should be the same as the body joint number
2. adjust the location of cubes based on the OFFSET and rotation data



Putting cubes is not hard; you can find useful functions in the provided example. Building a skeleton with OFFSET has been solved well by all students; the new position of the joint can be calculated by adding the position of the parent joint and OFFSET. A for loop in parent variables will manage it well.

```python
current_joint_position = parent_joint_position + OFFSET 
```



The main challenge is the FK animation part; how can we drive these cubes based on their rotation? Two ideas can be shared. 

<img width="611" alt="20220318134647" src="https://user-images.githubusercontent.com/7709951/158945342-034cda32-07ac-4eb4-818f-d3ddacb0e979.png">

I will give the explanation based on an example with a three-joint chain, located initially in (1, 1), (2, 1), and (3, 1). We will apply 45 degrees on each joint.

#### FK - Rotation by Rotation

The straightforward idea is to consider the whole process of rotation by rotation. Firstly we calculate the final status after applying the first rotation, get the new position of all child joints, and then calculate the new position based on the updated joint position. 

The green line means the rotated bones applied on the first rotation of joint 0. So the position of joint 1 and joint 2 will be updated. 

The red line means the rotated bones applied on the first rotation of joint 1.  So the position of joint 2 will also be updated.

<img width="746" alt="20220318135144" src="https://user-images.githubusercontent.com/7709951/158945335-e1de22cd-7674-45b8-af72-1d8fe2ca2b59.png">

Some students use this idea in their code, but because many variables should be updated each time, they mess things up to yield wrong results. Also, for each rotation, the position of all joints will be updated at least once in the calculation, so the complexity is about n·log(n). 



#### FK - Joint by Joint

I will recommend this idea more because it reduces the complexity and is easier to implement.

In each step, we only consider the final status of **one** joint. 

For example, we first consider joint1; it will be affected only by one parent joint - joint 0, so we apply the rotation of joint0 on the connected bone. So the final position of joint 1 will be: parent_position + bone0 @ 45 degree. 

Then, we consider the joint2. Two parent joints will affect it, so we **accumulate** their rotations: 45+45 = 90 and apply the accumulated rotation on connected bone. And this result will be the final status of joint2.

<img width="706" alt="20220318140607" src="https://user-images.githubusercontent.com/7709951/158946744-22c77489-4647-4d50-8660-0e57438e3665.png">

Noticeable here, the position of joint2 was only updated once. So the complexity will be n. Also, this method avoids multiply updates for one joint, so it can also help you debug and organize. 



#### Example Code

```python
def add_cube(location, name):
    bpy.ops.mesh.primitive_cube_add(size=0.5, enter_editmode=False, location=location)
    new_cube = bpy.context.object
    new_cube.name = name
    return new_cube

def build_joint_cubes(offsets, parents, names, joint, parent_obj, all_obj):
    if joint != 0:
        offset = mathutils.Vector(offsets[joint])
        new_cube = add_cube(parent_obj.location + offset, names[joint] + '_cube')
    else:
        new_cube = add_cube(mathutils.Vector((0., 0., 0.)), names[joint] + '_cube')
    all_obj.append(new_cube)

    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx == joint:
            build_joint_cubes(offsets, parents, names, joint_idx, new_cube, all_obj)
            
def set_cube_animation(joints, rotations, parents, positions, offsets,frametime):
    global_position = np.zeros((rotations.shape[0], rotations.shape[1], 3))
    
    # A build-in function in Quaternion class
    transforms = rotations.transforms()
    root_joint_position = positions[:, 0]
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx == -1:
            global_position[:, joint_idx] = root_joint_position
            continue
        rotated_vector = np.matmul(transforms[:, parent_idx], mathutils.Vector(offsets[joint_idx]))
        global_position[:, joint_idx] = global_position[:, parent_idx] + rotated_vector
        transforms[:, joint_idx] = np.matmul(transforms[:, parent_idx], transforms[:, joint_idx])
    
    # Insert keyframes
    for frame_idx in range(rotations.shape[0]):
        for joint_idx,parent_idx in enumerate(parents):
            joints[joint_idx].location = global_position[frame_idx, joint_idx]
            joints[joint_idx].keyframe_insert(data_path='location', frame=frame_idx)
            
all_obj = []
build_joint_cubes(offsets, parents, names, 0, None, all_obj)
set_cube_animation(all_obj, rotations, parents, positions, offsets, frametime)
```



##Score Distribution

<img width="354" alt="20220318144600" src="https://user-images.githubusercontent.com/7709951/158950976-ec6456dd-f861-42f1-b7c7-b67a9d459764.png">

Some students missed some files in the submission; I left comments on Moodle Grade System; pls check it and argue it with me. 

And again, these assignments will be challenging if you have no any experiences with computer graphics, and our marks will be considered based on this situation. Always keep your idea in the report file, so we can know your achievements and give scores for them. 