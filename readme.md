# simple VAE in PyTorch

use a encoder to map a image into a laten, then use MLP map laten to mean and log var, and accordingly sample noise, finally reconstuct original image from noise.

this code can sample a decent image on mnist data, but seem to fail in a more complex datasets like flower dataset.


# Results
## mnist
1. sampled image

![img](assets/vae_sampled_mnist.png)

1. reconstructed image

![img](assets/vae_reconstructed_mnist.png)

## flower dataset

### when reg loss w = 0.1
1. sampled image

![img](assets/vae-reg-w-0.1-sample.png)

1. reconstructed image

![img](assets/vae-reg-w-0.1-reconst.png)

### when reg loss w = 0.01
1. sampled image

![img](assets/vae-reg-w-0.01-sample.png)

1. reconstructed image

![img](assets/vae-reg-w-0.01-reconst.png)

### when reg loss w = 0.0001
1. sampled image

![img](assets/vae-reg-w-0.0001-sample.png)

1. reconstructed image

![img](assets/vae-reg-0.0001-reconst.png)
