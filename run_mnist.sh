device_idx=7
config_p='configs/config_mnist.py'
run_name='reg-w-0.075'
reg_loss_w=0.075

CUDA_VISIBLE_DEVICES=$device_idx  python run.py \
                        --config $config_p \
                        --run_name $run_name \
                        --reg_loss_w $reg_loss_w