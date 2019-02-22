call activate tf-gpu
if not exist "ir\" mkdir ir
call python C:\Intel\computer_vision_sdk\deployment_tools\model_optimizer\mo_tf.py --input_model frozen_yolow.pb --tensorflow_use_custom_operations_config yolow_ir_config.json --batch 1 --data_type FP16 --output_dir ir\
call C:\Intel\computer_vision_sdk\bin\setupvars.bat