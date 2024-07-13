device = 'cuda'

num_ep = 100
optimizer_config = dict(
    type='adamw',
    config = dict(
        lr = 3e-4,
    )
)

lr_sche_config = dict(
    type = 'constant',
    config = dict(
        # warm_up_epoch=1
    )
)



####---- model ----####
img_size = 32
dataset_type = 'mnist'
model_config = dict(
    encoder_config = dict(
                 in_channel=1, 
                 init_channels=16,
                 channels_lst=[16, 32, 32, 64],
                 num_groups_lst=[4, 4, 4, 4],
                 num_res_layers_per_resolution=2,
                 layer_with_attention=[False, False, False, False],
    ),
    decoder_config = dict(
                 out_channel=1,
                 channels_lst=[64, 32, 32, 16],
                 num_groups_lst=[4, 4, 4, 4],
                 num_res_layers_per_resolution=2,
                 layer_with_attention=[False, False, False, False],
    ),
    laten_dim = 128,
    device = device,
    img_size = img_size,
    reg_loss_w = 0.1
)

####---- model ----####



####---- data ----####
data_root = '/home/dmt/shao-tao-working-dir/DATA/OpenDataLab___Oxford_102_Flower/raw/jpg'
train_data_config = dict(
    transform_config = dict(
        img_size = img_size,
        mean=(0.5, ),
        std=(0.5, )
    ),
    dataset_config = dict(
        root='.'
    ), 
    data_loader_config = dict(
        batch_size = 16,
        num_workers = 2,
    )
)
val_data_config = dict(
    transform_config = dict(
        img_size = img_size,
        mean=(0.5, ),
        std=(0.5, )
    ),
    dataset_config = dict(
        root='.'
    ), 
    data_loader_config = dict(
        batch_size = 16,
        num_workers = 2,
    )
)
####---- data ----####


resume_ckpt_path = None
load_weight_from = None

# ckp
ckp_config = dict(
   save_last=True, 
   every_n_epochs=None,
#    monitor='val_mae',
#    mode='min',
#    filename='{epoch}-{val_mae:.3f}'
)

# trainer config
trainer_config = dict(
    log_every_n_steps=5,
    precision='32',
    # val_check_interval=1, # val after k training batch 0.0-1.0, or a int
    check_val_every_n_epoch=1,
    num_sanity_val_steps=2
)


# LOGGING
enable_wandb = True
wandb_config = dict(
    project = 'vae',
    offline = True
)
ckp_root = f'[{wandb_config["project"]}]'