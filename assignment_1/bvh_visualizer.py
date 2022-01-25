# -*- coding: utf-8 -*-
# Copyright 2022 by HKU-COMP3360-Data-Driven-Animation
# Author: myshi@cs.hku.hk, taku@cs.hku.hk
# Thanks to Daniel Holden, Peizhuo Li

import re
import bpy
import mathutils
import numpy as np

class Quaternions:
    """
    Quaternions is a wrapper around a numpy ndarray
    that allows it to act as if it were an narray of
    a quater data type.
    
    Therefore addition, subtraction, multiplication,
    division, negation, absolute, are all defined
    in terms of quater operations such as quater
    multiplication.
    
    This allows for much neater code and many routines
    which conceptually do the same thing to be written
    in the same way for point data and for rotation data.
    
    The Quaternions class has been desgined such that it
    should support broadcasting and slicing in all of the
    usual ways.
    """
    
    def __init__(self, qs):
        if isinstance(qs, np.ndarray):
            if len(qs.shape) == 1: qs = np.array([qs])
            self.qs = qs
            return

        if isinstance(qs, Quaternions):
            self.qs = qs
            return

        raise TypeError('Quaternions must be constructed from iterable, numpy array, or Quaternions, not %s' % type(qs))
    
    def __str__(self): return "Quaternions("+ str(self.qs) + ")"
    def __repr__(self): return "Quaternions("+ repr(self.qs) + ")"
    
    """ Helper Methods for Broadcasting and Data extraction """
    
    @classmethod
    def _broadcast(cls, sqs, oqs, scalar=False):
        if isinstance(oqs, float): return sqs, oqs * np.ones(sqs.shape[:-1])
        
        ss = np.array(sqs.shape) if not scalar else np.array(sqs.shape[:-1])
        os = np.array(oqs.shape)

        if len(ss) != len(os):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))
            
        if np.all(ss == os): return sqs, oqs
        
        if not np.all((ss == os) | (os == np.ones(len(os))) | (ss == np.ones(len(ss)))):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))

        sqsn, oqsn = sqs.copy(), oqs.copy()

        for a in np.where(ss == 1)[0]: sqsn = sqsn.repeat(os[a], axis=a)
        for a in np.where(os == 1)[0]: oqsn = oqsn.repeat(ss[a], axis=a)
        
        return sqsn, oqsn
        
    """ Adding Quaterions is just Defined as Multiplication """
    
    def __add__(self, other): return self * other
    def __sub__(self, other): return self / other
    
    """ Quaterion Multiplication """
    
    def __mul__(self, other):
        """
        Quaternion multiplication has three main methods.
        
        When multiplying a Quaternions array by Quaternions
        normal quater multiplication is performed.
        
        When multiplying a Quaternions array by a vector
        array of the same shape, where the last axis is 3,
        it is assumed to be a Quaternion by 3D-Vector 
        multiplication and the 3D-Vectors are rotated
        in space by the Quaternions.
        
        When multipplying a Quaternions array by a scalar
        or vector of different shape it is assumed to be
        a Quaternions by Scalars multiplication and the
        Quaternions are scaled using Slerp and the identity
        quaternions.
        """
        
        """ If Quaternions type do Quaternions * Quaternions """
        if isinstance(other, Quaternions):
            sqs, oqs = Quaternions._broadcast(self.qs, other.qs)

            q0 = sqs[...,0]; q1 = sqs[...,1]; 
            q2 = sqs[...,2]; q3 = sqs[...,3]; 
            r0 = oqs[...,0]; r1 = oqs[...,1]; 
            r2 = oqs[...,2]; r3 = oqs[...,3]; 
            
            qs = np.empty(sqs.shape)
            qs[...,0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
            qs[...,1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
            qs[...,2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
            qs[...,3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
            
            return Quaternions(qs)
        
        """ If array type do Quaternions * Vectors """
        if isinstance(other, np.ndarray) and other.shape[-1] == 3:
            vs = Quaternions(np.concatenate([np.zeros(other.shape[:-1] + (1,)), other], axis=-1))

            return (self * (vs * -self)).imaginaries

        """ If float do Quaternions * Scalars """
        if isinstance(other, np.ndarray) or isinstance(other, float):
            return Quaternions.slerp(Quaternions.id_like(self), self, other)
        
        raise TypeError('Cannot multiply/add Quaternions with type %s' % str(type(other)))
        
    def __div__(self, other):
        """
        When a Quaternion type is supplied, division is defined
        as multiplication by the inverse of that Quaternion.
        
        When a scalar or vector is supplied it is defined
        as multiplicaion of one over the supplied value.
        Essentially a scaling.
        """
        
        if isinstance(other, Quaternions): return self * (-other)
        if isinstance(other, np.ndarray): return self * (1.0 / other)
        if isinstance(other, float): return self * (1.0 / other)
        raise TypeError('Cannot divide/subtract Quaternions with type %s' + str(type(other)))
        
    def __eq__(self, other): return self.qs == other.qs
    def __ne__(self, other): return self.qs != other.qs
    
    def __neg__(self):
        """ Invert Quaternions """
        return Quaternions(self.qs * np.array([[1, -1, -1, -1]]))
    
    def __abs__(self):
        """ Unify Quaternions To Single Pole """
        qabs = self.normalized().copy()
        top = np.sum(( qabs.qs) * np.array([1,0,0,0]), axis=-1)
        bot = np.sum((-qabs.qs) * np.array([1,0,0,0]), axis=-1)
        qabs.qs[top < bot] = -qabs.qs[top <  bot]
        return qabs
    
    def __iter__(self): return iter(self.qs)
    def __len__(self): return len(self.qs)
    
    def __getitem__(self, k):    return Quaternions(self.qs[k]) 
    def __setitem__(self, k, v): self.qs[k] = v.qs
        
    @property
    def lengths(self):
        return np.sum(self.qs**2.0, axis=-1)**0.5
    
    @property
    def reals(self):
        return self.qs[...,0]
        
    @property
    def imaginaries(self):
        return self.qs[...,1:4]
    
    @property
    def shape(self): return self.qs.shape[:-1]
    
    def repeat(self, n, **kwargs):
        return Quaternions(self.qs.repeat(n, **kwargs))
    
    def normalized(self):
        return Quaternions(self.qs / self.lengths[...,np.newaxis])
    
    def log(self):
        norm = abs(self.normalized())
        imgs = norm.imaginaries
        lens = np.sqrt(np.sum(imgs**2, axis=-1))
        lens = np.arctan2(lens, norm.reals) / (lens + 1e-10)
        return imgs * lens[...,np.newaxis]
    
    def constrained(self, axis):
        
        rl = self.reals
        im = np.sum(axis * self.imaginaries, axis=-1)
        
        t1 = -2 * np.arctan2(rl, im) + np.pi
        t2 = -2 * np.arctan2(rl, im) - np.pi
        
        top = Quaternions.exp(axis[np.newaxis] * (t1[:,np.newaxis] / 2.0))
        bot = Quaternions.exp(axis[np.newaxis] * (t2[:,np.newaxis] / 2.0))
        img = self.dot(top) > self.dot(bot)
        
        ret = top.copy()
        ret[ img] = top[ img]
        ret[~img] = bot[~img]
        return ret
    
    def constrained_x(self): return self.constrained(np.array([1,0,0]))
    def constrained_y(self): return self.constrained(np.array([0,1,0]))
    def constrained_z(self): return self.constrained(np.array([0,0,1]))
    
    def dot(self, q): return np.sum(self.qs * q.qs, axis=-1)
    
    def copy(self): return Quaternions(np.copy(self.qs))
    
    def reshape(self, s):
        self.qs.reshape(s)
        return self
    
    def interpolate(self, ws):
        return Quaternions.exp(np.average(abs(self).log, axis=0, weights=ws))
    
    def euler(self, order='xyz'):
        
        q = self.normalized().qs
        q0 = q[...,0]
        q1 = q[...,1]
        q2 = q[...,2]
        q3 = q[...,3]
        es = np.zeros(self.shape + (3,))
        
        if   order == 'xyz':
            es[..., 2] = np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
            es[..., 1] = np.arcsin((2 * (q1 * q3 + q0 * q2)).clip(-1,1))
            es[..., 0] = np.arctan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        else:
            raise NotImplementedError('Cannot convert from ordering %s' % order)
        
        return es
        
    
    def average(self):
        
        if len(self.shape) == 1:
            
            import numpy.core.umath_tests as ut
            system = ut.matrix_multiply(self.qs[:,:,np.newaxis], self.qs[:,np.newaxis,:]).sum(axis=0)
            w, v = np.linalg.eigh(system)
            qiT_dot_qref = (self.qs[:,:,np.newaxis] * v[np.newaxis,:,:]).sum(axis=1)
            return Quaternions(v[:,np.argmin((1.-qiT_dot_qref**2).sum(axis=0))])            
        
        else:
            
            raise NotImplementedError('Cannot average multi-dimensionsal Quaternions')

    def angle_axis(self):
        
        norm = self.normalized()        
        s = np.sqrt(1 - (norm.reals**2.0))
        s[s == 0] = 0.001
        
        angles = 2.0 * np.arccos(norm.reals)
        axis = norm.imaginaries / s[...,np.newaxis]
        
        return angles, axis
        
    
    def transforms(self):
        
        qw = self.qs[...,0]
        qx = self.qs[...,1]
        qy = self.qs[...,2]
        qz = self.qs[...,3]
        
        x2 = qx + qx; y2 = qy + qy; z2 = qz + qz;
        xx = qx * x2; yy = qy * y2; wx = qw * x2;
        xy = qx * y2; yz = qy * z2; wy = qw * y2;
        xz = qx * z2; zz = qz * z2; wz = qw * z2;

        m = np.empty(self.shape + (3,3))
        m[...,0,0] = 1.0 - (yy + zz)
        m[...,0,1] = xy - wz
        m[...,0,2] = xz + wy
        m[...,1,0] = xy + wz
        m[...,1,1] = 1.0 - (xx + zz)
        m[...,1,2] = yz - wx
        m[...,2,0] = xz - wy
        m[...,2,1] = yz + wx
        m[...,2,2] = 1.0 - (xx + yy)
        
        return m
    
    def ravel(self):
        return self.qs.ravel()
    
    @classmethod
    def id(cls, n):
        
        if isinstance(n, tuple):
            qs = np.zeros(n + (4,))
            qs[...,0] = 1.0
            return Quaternions(qs)
        
        if isinstance(n, int):
            qs = np.zeros((n,4))
            qs[:,0] = 1.0
            return Quaternions(qs)
        
        raise TypeError('Cannot Construct Quaternion from %s type' % str(type(n)))

    @classmethod
    def id_like(cls, a):
        qs = np.zeros(a.shape + (4,))
        qs[...,0] = 1.0
        return Quaternions(qs)
        
    @classmethod
    def exp(cls, ws):
    
        ts = np.sum(ws**2.0, axis=-1)**0.5
        ts[ts == 0] = 0.001
        ls = np.sin(ts) / ts
        
        qs = np.empty(ws.shape[:-1] + (4,))
        qs[...,0] = np.cos(ts)
        qs[...,1] = ws[...,0] * ls
        qs[...,2] = ws[...,1] * ls
        qs[...,3] = ws[...,2] * ls
        
        return Quaternions(qs).normalized()
        
    @classmethod
    def slerp(cls, q0s, q1s, a):
        
        fst, snd = cls._broadcast(q0s.qs, q1s.qs)
        fst, a = cls._broadcast(fst, a, scalar=True)
        snd, a = cls._broadcast(snd, a, scalar=True)
        
        len = np.sum(fst * snd, axis=-1)
        
        neg = len < 0.0
        len[neg] = -len[neg]
        snd[neg] = -snd[neg]
        
        amount0 = np.zeros(a.shape)
        amount1 = np.zeros(a.shape)

        linear = (1.0 - len) < 0.01
        omegas = np.arccos(len[~linear])
        sinoms = np.sin(omegas)
        
        amount0[ linear] = 1.0 - a[linear]
        amount1[ linear] =       a[linear]
        amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms
        amount1[~linear] = np.sin(       a[~linear]  * omegas) / sinoms
        
        return Quaternions(
            amount0[...,np.newaxis] * fst + 
            amount1[...,np.newaxis] * snd)
    
    @classmethod
    def between(cls, v0s, v1s):
        a = np.cross(v0s, v1s)
        w = np.sqrt((v0s**2).sum(axis=-1) * (v1s**2).sum(axis=-1)) + (v0s * v1s).sum(axis=-1)
        return Quaternions(np.concatenate([w[...,np.newaxis], a], axis=-1)).normalized()
    
    @classmethod
    def from_angle_axis(cls, angles, axis):
        axis    = axis / (np.sqrt(np.sum(axis**2, axis=-1)) + 1e-10)[...,np.newaxis]
        sines   = np.sin(angles / 2.0)[...,np.newaxis]
        cosines = np.cos(angles / 2.0)[...,np.newaxis]
        return Quaternions(np.concatenate([cosines, axis * sines], axis=-1))
    
    @classmethod
    def from_euler(cls, es, order='xyz', world=False):
    
        axis = {
            'x' : np.array([1,0,0]),
            'y' : np.array([0,1,0]),
            'z' : np.array([0,0,1]),
        }
        
        q0s = Quaternions.from_angle_axis(es[...,0], axis[order[0]])
        q1s = Quaternions.from_angle_axis(es[...,1], axis[order[1]])
        q2s = Quaternions.from_angle_axis(es[...,2], axis[order[2]])
        
        return (q2s * (q1s * q0s)) if world else (q0s * (q1s * q2s))
    
    @classmethod
    def from_transforms(cls, ts):
        
        d0, d1, d2 = ts[...,0,0], ts[...,1,1], ts[...,2,2]
        
        q0 = ( d0 + d1 + d2 + 1.0) / 4.0
        q1 = ( d0 - d1 - d2 + 1.0) / 4.0
        q2 = (-d0 + d1 - d2 + 1.0) / 4.0
        q3 = (-d0 - d1 + d2 + 1.0) / 4.0
        
        q0 = np.sqrt(q0.clip(0,None))
        q1 = np.sqrt(q1.clip(0,None))
        q2 = np.sqrt(q2.clip(0,None))
        q3 = np.sqrt(q3.clip(0,None))
        
        c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3)
        c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
        c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3)
        c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)
        
        q1[c0] *= np.sign(ts[c0,2,1] - ts[c0,1,2])
        q2[c0] *= np.sign(ts[c0,0,2] - ts[c0,2,0])
        q3[c0] *= np.sign(ts[c0,1,0] - ts[c0,0,1])
        
        q0[c1] *= np.sign(ts[c1,2,1] - ts[c1,1,2])
        q2[c1] *= np.sign(ts[c1,1,0] + ts[c1,0,1])
        q3[c1] *= np.sign(ts[c1,0,2] + ts[c1,2,0])  
        
        q0[c2] *= np.sign(ts[c2,0,2] - ts[c2,2,0])
        q1[c2] *= np.sign(ts[c2,1,0] + ts[c2,0,1])
        q3[c2] *= np.sign(ts[c2,2,1] + ts[c2,1,2])  
        
        q0[c3] *= np.sign(ts[c3,1,0] - ts[c3,0,1])
        q1[c3] *= np.sign(ts[c3,2,0] + ts[c3,0,2])
        q2[c3] *= np.sign(ts[c3,2,1] + ts[c3,1,2])  
        
        qs = np.empty(ts.shape[:-2] + (4,))
        qs[...,0] = q0
        qs[...,1] = q1
        qs[...,2] = q2
        qs[...,3] = q3
        
        return cls(qs)


channelmap = {
    'Xrotation' : 'x',
    'Yrotation' : 'y',
    'Zrotation' : 'z'   
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x' : 0,
    'y' : 1,
    'z' : 2,
}

def load(filename, start=None, end=None, order=None, world=False):
    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False
    
    names = []
    offsets = np.array([]).reshape((0,3))
    parents = np.array([], dtype=int)
    
    for line in f:
        
        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets    = np.append(offsets,    np.array([[0,0,0]]),   axis=0)
            parents    = np.append(parents, active)
            active = (len(parents)-1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site: end_site = False
            else: active = parents[active]
            continue
        
        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue
           
        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2+channelis:2+channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets    = np.append(offsets,    np.array([[0,0,0]]),   axis=0)
            parents    = np.append(parents, active)
            active = (len(parents)-1)
            continue
        
        if "End Site" in line:
            end_site = True
            continue
              
        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start)-1
            else:
                fnum = int(fmatch.group(1))
            jnum = len(parents)
            # result: [fnum, J, 3]
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            # result: [fnum, len(orients), 3]
            rotations = np.zeros((fnum, jnum, 3))
            continue
        
        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue
        
        if (start and end) and (i < start or i >= end-1):
            i += 1
            continue
        
        dmatch = line.strip().split()
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if   channels == 3:
                # This should be root positions[0:1] & all rotations
                positions[fi,0:1] = data_block[0:3]
                rotations[fi, : ] = data_block[3: ].reshape(N,3)
            elif channels == 6:
                data_block = data_block.reshape(N,6)
                # fill in all positions
                positions[fi,:] = data_block[:,0:3]
                rotations[fi,:] = data_block[:,3:6]
            elif channels == 9:
                positions[fi,0] = data_block[0:3]
                data_block = data_block[3:].reshape(N-1,9)
                rotations[fi,1:] = data_block[:,3:6]
                positions[fi,1:] += data_block[:,0:3] * data_block[:,6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()
    rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)
    return rotations, positions, offsets, parents, names, frametime

def clear_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def add_bone(offset, parent_obj, name):
    center = parent_obj.location + offset / 2
    length = offset.dot(offset) ** 0.5

    base = mathutils.Vector((0., 0., 1.))
    target = offset.normalized()
    axis = base.cross(target)
    theta = np.math.acos(base.dot(target))
    rot = mathutils.Quaternion(axis, theta)

    bpy.ops.mesh.primitive_cone_add(vertices=5, radius1=0.022 * 10, radius2=0.0132 * 10, depth=length, enter_editmode=False, location=center)
    new_bone = bpy.context.object
    new_bone.name = name
    new_bone.rotation_mode = 'QUATERNION'
    new_bone.rotation_quaternion = rot

    set_parent(parent_obj, new_bone)

    return new_bone

def add_joint(location, parent_obj, name):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.001, enter_editmode=False, location=location)
    new_joint = bpy.context.object
    if parent_obj is not None:
        set_parent(parent_obj, new_joint)
        pass
    new_joint.name = name

    return new_joint

def set_parent(parent, child):
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()

## Example - Use rotation and location as keyframe feature
def build_reset_skeleton(offsets, parents, names, joint, parent_obj, all_obj):
    if joint != 0:
        offset = mathutils.Vector(offsets[joint])
        new_bone = add_bone(offset, parent_obj, names[joint] + '_bone')
        new_joint = add_joint(parent_obj.location + offset, new_bone, names[joint] + '_end')
        all_obj.append(new_bone)
        all_obj.append(new_joint)
    else:
        new_joint = add_joint(mathutils.Vector((0., 0., 0.)), None, names[joint] + '_end')
        all_obj.append(new_joint)

    for i in range(len(parents)):
        if parents[i] == joint:
            build_reset_skeleton(offsets, parents, names, i, new_joint, all_obj)

def set_animation(joints, positions, rotations, frametime):
    import math
    ## Your implement here
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = rotations.shape[0] - 1
    bpy.context.scene.render.fps = int(1 / frametime)
    for frame in range(rotations.shape[0]):
        joints[0].location = positions[frame, 0, :]
        joints[0].keyframe_insert(data_path='location', frame=frame)
        for j in range(rotations.shape[1]):
            joints[j].rotation_mode = 'QUATERNION'
            joints[j].rotation_quaternion = mathutils.Quaternion(rotations[frame, j])
            joints[j].keyframe_insert(data_path='rotation_quaternion', frame=frame)

'''
Your implement here: Convert offsets to global joint positions
To-Do:
Put cubes on the position of each joint, which can be caculated by OFFSET
'''
def build_joint_cubes(offsets, parents, names, parent_obj, all_obj):
    
    return 

'''
Your implement here: Use rotations to update global joint positions
To-Do:
Calculate the joint position by the hierarchical structure and joint rotation 
Imagine: 
    There is a root joint in [0, 0, 0] and a child joint by OFFSET [1.363060 -1.794630 0.839290],
    Then given a rotation on this connection, what will happened?
* The rotation on parent joint will impact all child joint
* Any rotation library is allowed to use, like scipy
* 
'''
def set_cube_animation(joints, rotations, parents):
    global_position = np.zeros((rotations.shape[0], rotations.shape[1], 3))
    ## Your implement here: Convert the rotations to positions
    return


clear_objects()

## rotation - Quaternion class
'''
filename should be replaced by your own absolute file path
rotations   -   Quaternion class
positions   -   np.array with shape (frame, joint_number, 3)
offsets     -   np.array with shape (frame, joint_number, 3)
parents     -   np.array with shape (joint_number)
'''
rotations, positions, offsets, parents, names, frametime = load(filename='./assignment1/data/motion_files/motion_basket.bvh')
rotations.qs, positions, offsets = rotations.qs[..., [0, 3, 1, 2]], positions[..., [2, 0, 1]], offsets[..., [2, 0, 1]]

all_obj = []
build_reset_skeleton(offsets, parents, names, 0, None, all_obj)
all_joints = []
for j in range(rotations.shape[1]):
    name = names[j]
    for obj in all_obj:
        if obj.name == name + '_end':
            all_joints.append(obj)
set_animation(all_joints, positions, rotations.qs, frametime)
bpy.ops.object.select_all(action='DESELECT')
