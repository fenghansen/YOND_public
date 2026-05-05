# You Only Need a Denoiser  
[![arXiv](https://img.shields.io/badge/arXiv-2506.03645-b31b1b.svg)](https://arxiv.org/abs/2506.03645)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces/hansen97/YOND)
[![中文简介](https://img.shields.io/badge/中文简介-VMCL-0078d4)](https://vmcl-isp.site/t/topic/201)

The official implementation of *"[YOND: Practical Blind Raw Image Denoising Free from Camera-Specific Data Dependency](https://arxiv.org/abs/2506.03645)"*  
## Notes  
1. **Code Availability**  
   - YOND's core modules (CNE & EM-VST) have been **temporarily obfuscated with PyArmor** to comply with recent laboratory confidentiality regulations.  
   - You can still **run inference**, but the source code will remain encrypted **until the paper is officially accepted**.  

2. **Try YOND Online – No Installation Needed**
   - Upload your own noisy raw images and let YOND denoise them with our Hugging Face Spaces demo, powered by the ZeroGPU grant.  
   - 👉 **Demo & Inference:** [huggingface.co/spaces/hansen97/YOND](https://huggingface.co/spaces/hansen97/YOND)  

4. Complete experimental results (visualized as RGB images) are available at [Baidu Netdisk](https://pan.baidu.com/s/1tAJW_57v2jWJ605MWKMKVg?pwd=nn4v). This includes YOND's inference results on four public datasets (`results`) and crops used for comparison in the manuscript (`crops@paper`).  

5. Please download the datasets to the same directory and update the `host` in the `get_host_with_dir()` function in ```utils/utils.py```.  
  
## Training  
We have provided the pre-trained SNR-Net weights (`checkpoints`) and training data (`datasets/YOND`) at [Baidu Netdisk](https://pan.baidu.com/s/1tAJW_57v2jWJ605MWKMKVg?pwd=nn4v).  
Our training data consists of crops from the ground truth of [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [SID Sony dataset](https://cchen156.github.io/SID.html). For SID dataset, 16-bit images were generated using *rawpy*. We have made the cropped version (datasets) available to help you align with our experiments.  
RGB images will be unprocessed into pseudo-raw images with random Bayer patterns for training the AWGN Raw Denoiser.  
  
In fact, using only the DIV2K dataset can achieve results comparable to the paper on SIDD dataset. Training on higher-quality datasets (such as [LSDIR dataset](https://github.com/ofsoundof/LSDIR)) will yield better results in practice.  
  
We encourage retraining for verification, as training SNR-Net is straightforward.  
You can also replace SNR-Net by using any prepared AWGN ***Raw*** denoiser. Simply write the network architecture in the appropriate format under `archs` and modify the yaml files under `runfiles/Gaussian` accordingly.  
  
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
python YOND_DND.py -f runfiles/YOND/DND_simple+full_pre_grumix.yml -m evaltest  
# ELD (Cleaning up...)  
python YOND_ELD.py -f runfiles/YOND/ELD_simple+full_pre_grumix.yml -m eval  
# LRID (Cleaning up...)  
python YOND_LRID.py -f runfiles/YOND/LRID_simple+full_pre_grumix.yml -m evaltest
```

## 🏷️ Citation
If you find our code helpful in your research or work please cite our paper.
```bibtex
@article{feng2025yond,
  title={YOND: Practical Blind Raw Image Denoising Free from Camera-Specific Data Dependency},
  author={Feng, Hansen and Wang, Lizhi and Huang, Yiqi and Li, Tong and Zhu, Lin and Huang, Hua},
  journal={arXiv preprint arXiv:2506.03645},
  year={2025}
}
```

## 📧 Contact
If you would like to get in-depth help from me, please feel free to contact me (hansen97@outlook.com / fenghansen@bit.edu.cn) with a brief self-introduction (including your name, affiliation, and position).
