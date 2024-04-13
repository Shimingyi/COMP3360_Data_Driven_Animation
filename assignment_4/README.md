# Assignment 4

## Submission
### Due
- Apr. 26th, 23:59

### Submission Contents

File format: A compressed file **[uid_name_assignment3.zip]** with:

1. your `rigid_body_dynamic.py`
2. a recorded video of your simulator
3. your report `uid_name_3.pdf`
4. your own mesh.obj (if any)

## Introduction
In this assignment, you need to implement a basic PBD simulator based on Jacobi approach,
which offers a parallelizable method for solving position-based constraints for a fast simulation.


## Examples
A successful implementation of the PBD simulator (with the default parameters) should produce a video similar to the following:

## Environment
Same as Assignment 3. If you want to save the video directly by Taichi, you need to install ffmpeg.

```bash
$ conda activate comp3360
$ conda install ffmpeg
```

## Task 1 - Basic PBD Simulator (80%)

You are required to implement **3** `TODO` in `cloth_pbd.py`. Please check the hints in the comments.


## Task 2 - Collision with a Mesh (Bonus)

Once you have implemented the basic simulator, there is an opportunity to enhance its functionality by adding extra features.
For example, support for collision with a mesh instead of a simple sphere. 
The idea is still similar to the sphere collision, i.e., detect if the point is inside the mesh and then project it out.
An easy way is first compute the vertex normal for the obstacle mesh.
Then, for every point of the cloth, find its closest point on the obstacle mesh.
If the displacement from the obstacle point to the cloth point have a negative dot product with the normal, then the point is inside the obstacle.
In this case, you can project the point out along the normal direction.

## Task 3 Report (20%)

- PDF format, no page size requirement, so you can also prepare it with PowerPoint or keynote or markdown
- The first two lines should introduce your NAME and UID.

You're expected to briefly introduce your implementation, and Task 2 if you have done it.
You also need to experiment with the parameters that will affect the material behavior, i.e., iteration number, alpha,
and analyze the results.