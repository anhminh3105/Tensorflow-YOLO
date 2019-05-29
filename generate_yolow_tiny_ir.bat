call activate tf-gpu
if not exist "data\ir\" mkdir data\ir
call python C:\Intel\computer_vision_sdk\deployment_tools\model_optimizer\mo_tf.py --input_model data\frozen_yolow_tiny.pb --tensorflow_use_custom_operations_config data\yolow_tiny_ir_config.json --batch 1 --data_type FP16 --scale 255 --output_dir data\ir\
call C:\Intel\computer_vision_sdk\bin\setupvars.bat
