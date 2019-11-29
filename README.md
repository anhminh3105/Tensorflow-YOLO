# TensorFlow YOLO (YOLOw)
A YOLOv3 implementation on TensorFlow core API

![webp net-gifmaker 3](https://user-images.githubusercontent.com/18170028/52903429-d9288180-3225-11e9-8249-ed435b234931.gif)

This is the thesis work of my bachelor degree at [VAMK](http://www.puv.fi/en/ "VAMK's Homepage").

**[Topic: Deploying YOLOv3 on Movidius Raspberry Pi](http://urn.fi/URN:NBN:fi:amk-2019103120519)**
## Summary:
* The model was written in TensorFlow Core API, data manipulation uses NumPy and OpenCV.
* High level programming interfaces `Yolow` offers quick and easy to load any (custom) Yolov3(-tiny) models.
* Allows to load Darknet trained models and predict on YOLOv3 model. (I'd be nice if someone would like to help me build the training pipeline, otherwise just do training in Darknet and load the trained model to YOLOw using `WeightLoader` in load.py)
* Easy to use `Imager` module to  handle image pre-processing, post-processing and visualisation.
* High-level Interface `YolowNCS` for running on Intel NCS (required installation of OpenVINO and converted models, please refer to OpenVINO workflow)
* Integratable to your applications.
* Code was written as modular as possible.


## Rooms to improve:
1. Training pipeline.

## Instructions:

### Dependencies:
- Python >=3.6.8
- TensorFlow/TensorFlow-GPU >=1.12
- OPENVINO toolkit version >=2018.0.5.456
- NumPy
- OpenCV2

### Installation:
any of these libraries above can be easily installed with the following command: `pip install package_name` except for TensorFlow-GPU as you have to also install Nvidia CUDA + cuDNN which is a very pessimistic hassle. Being committed to avoid this in every way, I use [Anaconda](https://www.anaconda.com/ "Anaconda Homepage") for ease of package and working environment management, which I will use for this instruction.

If you have never try to install Tensorflow-GPU in the prior way, I would recommend trying it first to have a taste to be able to appreciate the simplicity of the latter. ;)

1. download and install Anaconda from its Website.
   from this point I will be using my personal default of naming, you may change it however you like
2. Open Anaconda Prompt (or add the path of Anaconda Executable to sytem variables to use within CMD) and create a vitural environment named `tf-gpu`:
      ```
      conda create --name tf-gpu
      ```
3. activate it
      ```
      conda activate tf-gpu
      ```
4. Install TensorFlow-GPU (leave out the "-GPU" if you are installing Tensorflow non GPU version):
      ```
      conda install -c anaconda tensorflow-gpu
      ```
5. Install OpenCV2:
      ```
      conda install -c conda-forge opencv
      ```

### Usage:
* To detect on images, put them to path `data/images` and run the following command in prompt/conda prompt:
```
python detect_images.py -c data/config/<name_of_config_file>.cfg
```
* To detect live on camera, run the following command in prompt/conda prompt:
    ```
    python live.py -c data/config/<name_of_config_file>.cfg
    ```
### NCS:

#### Dependencies:
- OpenVINO
- picamera (optionally if you intend to run on rapberry pi equipped with Pi Camera module)
#### Installation:
*This instruction is in Windows but the procedures are the same for other OSes please follow up and refer to the installation instruction from Intel.
- Install [OpenVINO](https://software.intel.com/en-us/openvino-toolkit)
- if:

     1. you have set anaconda to system variables

     2. and used my naming for the virtual environment

     3. install openvino to C:\ drive.

  now run `generate_yolow_full_ir.bat` or `generate_yolow_tiny_ir.bat` in a command prompt (you may want to check it out and change it to suit your needs).

- else:

     1. Activate your virtual environment (skip if you dont use it)

     2. run the file C:\Intel\computer_vision_sdk\bin\setupvars.bat to source open-vino SDK for Python (or just run `activate_openvino_running_env.bat`)

     3. run the following command to generate the intermediate representations (IR):
         ```
         ...\YOLOw> python ...\Intel\computer_vision_sdk\deployment_tools\model_optimizer\mo_tf.py --input_model .\frozen_yolow.pb --tensorflow_use_custom_operations_config .\yolow_ir_config.json --batch 1 --data_type FP16 --output_dir .\ir\
         ```
 *Please also output the IR to the ir\ directory since the app will read from it by default.
#### Usage
- Start live demo on the NCS: 
  * Single NCS device
    ```
    python live_ncs.py -c data/config/<name_of_config_file>.cfg
    ```
  * NCS device)
    ```
    python live_multi_ncs.py -c data/config/<name_of_config_file>.cfg -n 2
    ```
## References:
[Original YOLO](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection)

[TensorFlow-SLIM YOLOv3](https://github.com/mystic123/tensorflow-yolo-v3)

[OpenVINO](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer)
