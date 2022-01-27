# -*- coding: utf-8 -*-
# Copyright 2022 by HKU-COMP3360-Data-Driven-Animation
# Author: myshi@cs.hku.hk, taku@cs.hku.hk
# Thanks to Archibate

import bpy
import numpy as np

## Clear all object
def clear_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

## Add a new cuda, refer to https://docs.blender.org/api/current/bpy.ops.mesh.html#bpy.ops.mesh.primitive_cube_add
def add_cube(location=[0, 0, 0], size=1):
    bpy.ops.mesh.primitive_cube_add(size=size)
    c_obj = bpy.context.object
    # c_obj = bpy.data.objects[bpy.context.active_object.name] # Another selection way
    c_obj.location = location
    c_obj.rotation_euler = [np.radians(0), np.radians(0), np.radians(0)]
    return c_obj

## Add a plane and materials
def add_floor(location=[0, 0, 0]):
    bpy.ops.mesh.primitive_plane_add(size=100, location=location)
    f_obj = bpy.context.object
    f_obj.name = 'floor'

    floor_mat = bpy.data.materials.new(name="floorMaterial")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
    floor_text = floor_mat.node_tree.nodes.new("ShaderNodeTexChecker")
    floor_text.inputs[3].default_value = 150
    floor_mat.node_tree.links.new(bsdf.inputs['Base Color'], floor_text.outputs['Color'])

    f_obj.data.materials.append(floor_mat)
    return f_obj

## Add a connected bones
def add_bones(joint1=[1,1,1], joint2=[1,1,1], joint3=[3,3,3]):
    bpy.ops.object.armature_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    b_obj = bpy.context.object
    
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    edit_bones = b_obj.data.edit_bones
    edit_bones.remove(b_obj.data.edit_bones[0]) # Remove the default bone
    b1 = edit_bones.new('bone1')
    b1.head = joint1
    b1.tail = joint2
    b2 = edit_bones.new('bone2')
    b2.head = joint2
    b2.tail = joint3
    edit_bones['bone2'].parent = edit_bones['bone1']

    bpy.ops.object.mode_set(mode='OBJECT')
    return b_obj

'''
skeleton example
     J5           # (0, 4)
      |             
J4 - J0 - J1       # (-3, 2), (0, 2), (3, 2)
     / \
   J3   J2         # (-2, 0), (2, 0)

joints = [[0, 2, 0], [3, 2, 0], [2, 0, 0], [-2, 0, 0], [-3, 2, 0], [0, 4, 0]]
parents = [-1, 0, 0, 0, 0, 0] # the index of parent joint, -1 means itself
We use the variable #parents# to represent the hierarchical structure
'''
def add_skeleton(joints, parents):
    assert(len(joints) == len(parents))
    bpy.ops.object.armature_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    b_obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    edit_bones = b_obj.data.edit_bones
    edit_bones.remove(b_obj.data.edit_bones[0])
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx == -1:
            continue
        b1 = edit_bones.new('name%s' % joint_idx)
        b1.head = joints[parent_idx]
        b1.tail = joints[joint_idx]
    bpy.ops.object.mode_set(mode='OBJECT')
    print(b_obj.keys())
    return b_obj

def add_joints(joints):
    joint_objs = []
    for joint in joints:
        joint_objs.append(add_cube(location=joint, size=0.25))
    return joint_objs

'''
Two step:
1. Calculate the joint position on each frame
2. Insert the new position as key frame
'''
def set_animation(target, start_positions, end_positions, frames=20):
    assert(len(start_positions) == len(target))
    assert(len(start_positions) == len(end_positions))
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = frames
    joint_number = len(target)
    start_positions, end_positions = np.array(start_positions, dtype=np.float32), np.array(end_positions, dtype=np.float32), 
    base_increment = (end_positions - start_positions)/frames
    for frame_idx in range(frames):
        new_position = start_positions + base_increment*frame_idx
        for joint_idx in range(joint_number):
            target[joint_idx].location = new_position[joint_idx]
            target[joint_idx].keyframe_insert(data_path='location', frame=frame_idx)

## A toy Animation example: Waving
def toy_animation():
    N = 128
    frame = 500
    locations = np.zeros((N, frame, 3))
    toy_objects = add_joints([[0, 0, 0]]*N)
    bpy.context.scene.frame_end = frame
    for frame_idx in range(frame):
        for i in range(N):
            toy_objects[i].location = [(i - N/2)/5, 0, np.sin(i*0.1 + frame_idx*0.05)]
            toy_objects[i].keyframe_insert(data_path='location', frame=frame_idx)


clear_objects()

# add_floor()
# add_cube([0, 0, 3])
# add_cube([1, 1, 3])
# add_bones()

# joints = [[0, 2, 0], [3, 2, 0], [2, 0, 0], [-2, 0, 0], [-3, 2, 0], [0, 4, 0]]
# parents = [-1, 0, 0, 0, 0, 0] # the index of parent joint, -1 means itself
# skeleton = add_skeleton(joints, parents)


# joints = [[0, 2, 0], [3, 2, 0], [2, 0, 0], [-2, 0, 0], [-3, 2, 0], [0, 4, 0]]
# end_positions = [[0, 0, 2], [3, 0, 2], [2, 0, 0], [-2, 0, 0], [-3, 0, 2], [0, 0, 4]]
# joints_objects = add_joints(joints=joints)
# set_animation(joints_objects, joints, end_positions)

toy_animation()