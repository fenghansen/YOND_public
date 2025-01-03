# You Only Need a Denoiser 
YOND: Practical Blind Raw Image Denoising Free from Camera-Specific Data Dependency
---
### Due to certain conflicts, we have temporarily taken down the Code (GitHub), and we will relaunch it upon completion of the cleanup (ETA: 2025.01.05).
**For evaluation, please download the datasets to the same folder and change the `host` at the function `get_host_with_dir()` in ```utils/utils.py```.**

## Train SNR-Net
We will provide the pre-trained SNR-Net weights later. Please download the `checkpoints` in this project.

We welcome retraining for verification since the training of SNR-Net is really simple.  

**Note:** The yaml files under `runfiles/Gaussian` should be modified to adapt your device.

1. Download the DIV2K dataset and run ```process.ipynb``` to divide dataset.
2. Train the model.  

```bash
## Train for evaluation 
# SNR-Net with black level clip (for SIDD, DND)
python trainer_DIV2K.py -f runfiles/Gaussian/GuidedResUnet_nf32_5to50_norm.yml -m train
# SNR-Net without black level clip (for ELD, LRID)
python trainer_DIV2K.py -f runfiles/Gaussian/GuidedResUnet_nf32_5to50_norm_noclip.yml -m train

## Optional
# UNet with black level clip (for ELD, LRID)
python trainer_DIV2K.py -f runfiles/Gaussian/Unet_5to50_norm.yml -m train
# UNet without black level clip (for ELD, LRID)
python trainer_DIV2K.py -f runfiles/Gaussian/Unet_5to50_norm_noclip.yml -m train
```

## Evaluation on public datasets

**Note:** The yaml files under `runfiles/YOND` should be modified to adapt your device.

```bash
# SIDD
python YOND_SIDD.py -f runfiles/YOND/SIDD_simple+full_pre_gru32n.yml -m eval
# DND
python YOND_DND.py -f runfiles/YOND/DND_simple+full_pre_gru32n.yml -m eval
# ELD
python YOND_ELD.py -f runfiles/YOND/ELD_simple+full_pre_gru32n.yml -m eval
# LRID
python YOND_LRID.py -f runfiles/YOND/LRID_simple+full_pre_gru32n.yml -m eval
```

## Evaluation on your raw images (to be continue)
Please refer to `YOND_any.py` to modify.
