call activate tf-gpu
if not exist "data\ir\" mkdir data\ir
call python C:\Intel\computer_vision_sdk\deployment_tools\model_optimizer\mo_tf.py --input_model data\pb\frozen_yolov3-tiny.pb --tensorflow_use_custom_operations_config data\config\yolov3_tiny_mo_config.json --batch 1 --data_type FP16 --scale 255 --output_dir data\ir\
call C:\Intel\computer_vision_sdk\bin\setupvars.bat
