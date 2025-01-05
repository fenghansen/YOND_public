# You Only Need a Denoiser  
The official implementation of *"YOND: Practical Blind Raw Image Denoising Free from Camera-Specific Data Dependency"*  
## Notes  
1. The full version of this project will be released soon after potential conflicts are resolved. The training code has not been fully tested after cleanup and may contain minor bugs.  
2. Complete experimental results (visualized as RGB images) are available at [Baidu Netdisk](https://pan.baidu.com/s/1nox4vMXwhpsIHG6qri5SIg?pwd=vmcl#list/path=%2F). This includes YOND's inference results on four public datasets and crops used for comparison in the manuscript.  
3. Please download the datasets to the same directory and update the `host` in the `get_host_with_dir()` function in ```utils/utils.py```.  
4. We plan to release a more user-friendly version in the future to enable denoising of arbitrary raw images. Stay tuned (possibly in February).  
  
## Training  
We have provided the pre-trained SNR-Net weights and training data at [Baidu Netdisk](https://pan.baidu.com/s/1nox4vMXwhpsIHG6qri5SIg?pwd=vmcl#list/path=%2F).  
Our training data consists of crops from the ground truth of DIV2K and SID. For SID, 16-bit images were generated using rawpy. We have made the cropped version (datasets) available to help you align your experimental data. RGB images are unprocessed into pseudo-raw images with random Bayer patterns for training the AWGN Raw Denoiser.  
  
In fact, using only the DIV2K dataset can achieve results comparable to the paper on SIDD dataset. Training on higher-quality datasets (such as LSDIR dataset) will yield better results in practice.  
  
We encourage retraining for verification, as training SNR-Net is straightforward.  
You can also replace SNR-Net by using any prepared AWGN **RAW** denoiser. Simply write the network architecture in the appropriate format under `archs` and modify the yaml files under `runfiles/Gaussian` accordingly.  
  
Once the above data and settings are ready, you only need to run the following commands:  
  
```bash  
## Train for evaluation   
# SNR-Net with black level clip (for SIDD, DND)  
python trainer_AWGN.py -f runfiles/Gaussian/GRU_5to50_norm_mix.yml -m train  
# SNR-Net without black level clip (for ELD, LRID)  
python trainer_AWGN.py -f runfiles/Gaussian/GRU_5to50_norm_mix_noclip.yml -m train  
  
## Training without SNR value guidance, i.e., YOND (UNet)  
python trainer_AWGN.py -f runfiles/Gaussian/Unet_5to50_norm_noclip.yml -m train  
```  
  
## Evaluation on Public Datasets  
**Note:** The yaml files under `runfiles/YOND` should be modified to match your device.  
  
```bash  
# SIDD  
python YOND_SIDD.py -f runfiles/YOND/SIDD_simple+full_pre_grumix.yml -m eval  
# DND (Cleaning up...)  
python YOND_DND.py -f runfiles/YOND/DND_simple+full_pre_grumix.yml -m eval  
# ELD (Cleaning up...)  
python YOND_ELD.py -f runfiles/YOND/ELD_simple+full_pre_grumix.yml -m eval  
# LRID (Cleaning up...)  
python YOND_LRID.py -f runfiles/YOND/LRID_simple+full_pre_grumix.yml -m eval  
```  
  
## Evaluation on Your Raw Images (To Be Continued)  
Please refer to `YOND_any.py` for modification (Cleaning up...).

