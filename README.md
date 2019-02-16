# YOLOw
A YOLOv3-416 implementation on TensorFlow Low-Level API

![webp net-gifmaker 3](https://user-images.githubusercontent.com/18170028/52903429-d9288180-3225-11e9-8249-ed435b234931.gif)

This is the thesis work I'm currently working on for my bachelor degree at [VAMK](http://www.puv.fi/en/ "VAMK's Homepage").

**Thesis topic: Deploying YOLOv3 on Movidius Raspberry Pi (3B+).**

I acknowledged that there are a lot of good works on YOLO already but since I am just so interested in this, and being also offered by my supervisor coincidentally, I just went on to make another version of my own.

## Summary:
* The model was made in pure TensorFlow Low-Level API, detection outputs were processed in NumPy and OpenCV2 was used for image manipulations.
* Non TensorFlow concise easy to use interface for loading, predicting on YOLOv3-416 model. (training to be added very soon)
* Also come with a concise easy to use interface for loading, pre-processing and post-processing images.
* Integratable to any applications.
* Code was written as modular as possible.

## What comes next:
1. Frozen model generation and handling, for needlessness of weight file loading every launch time.
2. Code to run model on Movidius Raspberry Pi.
3. Trainer for custom object detection.

## Instructions:

### Dependencies:
- TensorFlow/TensorFlow-GPU
- NumPy
- OpenCV2
- Jupyter Notebook (only pre-installed in Anaconda)

### Installation:
any of these libraries above can be easily installed with the following command: `pip install package_name` except for TensorFlow-GPU as you have to also install Nvidia CUDA + cuDNN which is a very pessimistic hassle. Being committed to avoid this in every way, I use [Anaconda](https://www.anaconda.com/ "Anaconda Homepage") for ease of package and working environment management, which I will use for this instruction. 

If you have never try to install Tensorflow-GPU in the prior way, I would recommend you to try it first to have a taste so you could appreciate the simplicity of the latter ;)

1. download and install Anaconda from its Website.
2. Open Anaconda Prompt/add the path of Anaconda Executable to variable path to use within CMD and issue command to create a new environment:
      ```
      conda create --name whatever_the_name_you_prefer
      ```
3. Now activate it
      ```
      conda activate the_env_name_you_just_created
      ```
4. Install TensorFlow-GPU (leave out the "-GPU" if you are installing Tensorflow non GPU version) *install either of them if you don't want to mess up your environment:
      ```
      conda install -c anaconda tensorflow-gpu
      ```
5. Install OpenCV2:
      ```
      conda install -c conda-forge opencv
      ```

### Usage:
* To detect on images, open file ***detect_images.ipynb*** with `jupyter notebook` and specify either the path to the image or the directory of images to the call of the method `imset_frompath(path)`, then run all the code. Output images will be saved to `outputs/` of the current working directory.
![image](https://user-images.githubusercontent.com/18170028/52904485-ab4b3900-3235-11e9-9a79-c23e94c1bf28.png)
* To detect live on camera, run the following command in prompt/conda prompt:
    ```
    python live.py
    ```
  
## References:
[Original YOLO](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection)

[DarkFlow](https://github.com/thtrieu/darkflow)

[TensorFlow-SLIM YOLOv3](https://github.com/mystic123/tensorflow-yolo-v3)
