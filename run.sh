device_idx=3
config_p='configs/config_flower.py'
reg_loss_w=0.01
run_name=flower-reg-w-${reg_loss_w}


CUDA_VISIBLE_DEVICES=$device_idx  python run.py \
                        --config $config_p \
                        --run_name $run_name \
                        --reg_loss_w $reg_loss_w