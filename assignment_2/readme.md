# Assignment 2 - Motion Processing with Deep Learning

![cover](https://user-images.githubusercontent.com/7709951/155896101-ba7aeed2-262d-40a0-9732-faee93d9b61b.gif)

Image copyright: Robust Motion In-betweening, Félix G. Harvey

## Introduction 

This assignment will teach you how to observe motion data and process it with different tools like interpolation algorithm and AutoEncoder model. Given a motion clip shaped with (T, J, R), understanding the difference between temporal dimensions and spatial dimensions will be the key to processing it; you will practice it by following tasks.

The example files are provided by @[Mingyi Shi](https://rubbly.cn/). Email him once you meet any problem. 



#### Tutorial Slides (2 Tutorials for each Assignment)

1. Introductory Motion Data Processing [[slides](../tutorial3_motion_processing.pdf)]
2. Data-Driven Motion Processing [[slides](../tutorial4_data_driven_motion_processing.pdf)]

#### Submission

Due to 11:59 am 18th, March**

File format: A compressed file which is named by uid_name_assignment1.zip, these files should be included:

- todo_motion_interpolation.py
- todo_motion_concatenation.py
- todo_ml_motion_denoising.py
- todo_ml_motion_interpolation.py
- report.pdf
- Results folder including BVH files
  - interpolation results (at least two)
  - concatenation results (at least two)
  - denoising results (one noised and one smoothed)
  - ml interpolation result



## Motion Processing - The Interpolation between keyframes (40%)

Interpolation is commonly used in the animation industry. For example, it happens when you control the character from standing state to walking state. A robust and high-quality interpolation algorithm will be helpful in this situation. Hence, in this task, you must implement two interpolation interfaces except for Linear Interpolation, which any 3rd library is allowed to use. 

**To-Do:**

1. Two template files: [todo_motion_interpolation.py](./todo_motion_interpolation.py) and [todo_motion_concatenation.py](./todo_motion_concatenation.py) 

   You should complete the apply_interpolation function with **two interpolation algorithms** and then test them in two subtasks - keyframe interpolation and motion concatenation. The implementation can be same in these two files, and any interpolation algorithm is allowed to use. 

2. Report: For the keyframe interpolation task, report the performance of different interpolation algorithms when given different [OFFSET](./todo_motion_interpolation.py#L38) values, like 24, 48, 72, 96, and 124. 

   1. Visual performance.  You can take screenshots and give the necessary explanation in your report to describing the difference between these algorithms when given different OFFSET. (10%)

   2. Comparison table. Calculate the per-joint positional difference between your interpolation result and ground truth, and then record the errors by different algorithms when given different OFFSET. (10%)

      ```
      error = np.sum((rotations_fake - rotations.qs)**2, axis=-1).mean()
      ```

3. Report: For the concatenation task, select the best interpolation interface in the previous task, and then try your best to shift the difference between the last frame of motion1 and the first frame of motion2. The score will be given by:

   1. Changing the [FRAME_INDEXs](./todo_motion_concatenation.py#L43) and reporting the visual performance on body rotation and global movement. You can take screenshots and give the necessary explanation in your report. (10%)
   2. Try to use some methods to shift the gap better when concatenating two motions. For example, the adaptation of root_joint_position, or maybe we can find two similar poses in motion one and motion 2, then use them as keyframes? Think about it and show your creativity and implementation. (10% for any trying) 

   

**Tips**: 

The last line is a call to open the Blender and display the generated results. You should config the blender execute file into your system path if you want to use it. You can follow these two links to get help: [macOS](https://blender.stackexchange.com/questions/2078/how-to-use-blender-command-lines-in-osx), [Windows](https://www.youtube.com/watch?v=n26RpbJgH_A). After trying, you can comment on this line and manually open the BVH results with Blender if you still struggle with the blender configuration. 

```python
subprocess.call('blender -P load_bvhs.py -- -r %s -c %s -o' % (output_file_path, output_file_path), shell=True)
```



## Data-Driven Motion Processing - Train the Motion AutoEncoder for different tasks  (60%)

In recent years, deep learning has grown fast in computer science. These data-driven methods can be more flexible to understand high-dimensional temporal and spatial information than the mathematics method. It can bring the knowledge from the existing dataset to solve a new problem. AutoEncoder\[[wiki](https://en.wikipedia.org/wiki/Autoencoder)\]\[[article](https://www.jeremyjordan.me/autoencoders/)\], as a primary network framework, makes huge success in various tasks like image processing/feature extracting and character animation tasks. 

In this part, two autoencoders will be exampled to show the basic structure and how it works for animation data. Then, you will be asked to find the solution to solve the motion denoising and interpolation problem with AutoEncoder. 

Recommended Reading: A deep learning framework for character motion synthesis and editing, ACM Transactions on Graphics 35(4), 2016 (Proceedings of SIGGRAPH 2016) \[[pdf](https://www.ipab.inf.ed.ac.uk/cgvu/motionsynthesis.pdf)\]

**To-do:**

1. Two template files: [todo_ml_motion_denoising.py](./todo_ml_motion_denoising.py) and [todo_ml_motion_interpolation.py](./todo_ml_motion_interpolation.py) 

   You should complete the **model structure definition** and complete the **model training and evaluation**. You can take any code snippets from example files to help you finish that or try to use other structures like RNN or Conv2D to solve it. Any other function or modification on the template file is also allowed. 

2. We provided the dataloader for the training. Basically, four variables will be token out per batch: [batch_rotations (B, T, J, R), batch_rotations_noised  (B, T, J, R), batch_root_positions (B, T, 3), file_name (B)]. You should set up two things: Firstly, how many frames you want to feed into the network in each forwarding, we [set it to 1](./ml_motion_autoencoder_fc.py#L121) for fully connected example, and 15 for convolution example; Secondly, how do you operate the joint rotation features, [flattening](./ml_motion_autoencoder_fc.py#L139) them to one dimensional is one option, you should think about it with your network design. 

3. For different tasks, we manipulate the network input in different ways:

   **Denoising (30%)**: We add gaussian noises on the input data, so you can set the network forwarding from noise data to clean data. 

   **Interpolation (30%)**: Given a clip of human motion, we mask some frames and hope the network can recover the original motion. There are different options to set input variables, we can send a whole motion clip that includes lots of zeros into the network as input, or we can only send the keyframes into the network. 

4. Report: For each task, there should be three contents to explain your idea and results:

   1. Visual performance: You can take screenshots and give a necessary explanation in your report to describing the results you observed. 
   2. Comparison table: Record the errors and losses(follow the calculation from example file) in different hyperparameters/training parameter/task conditions. The conditions represent the [noise_scaling](./todo_ml_motion_denoising.py#L10) and [between_frame](./todo_ml_motion_interpolation.py#L109) number. 


**Tips**:

- Because it's hard to prepare GPU support for each student, we reduce the training dataset to balance training quality and cost. Currently, the autoencoder_fc will be expected to take the 30s per epoch with CPU version PyTorch, and the convolutional version takes more. You can select the suitable version in your own environment. 
- Due to the limited training data, the current generated motion by exampled autoencoder is not very good, so that it will be seen as a baseline and comparable to your results.
- All the hyperparameters and training parameters can be modified by yourself. You can refer to this article to select the best value for you. For example, epoch_number can be small if you think it has been covered; [channel size](./ml_motion_autoencoder_conv1d.py#L27) can be small if the network runs slow.
