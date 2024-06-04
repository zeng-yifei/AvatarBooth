<h1>AvatarBooth: High-Quality and Customizable 3D Human Avatar Generation</h1>

<div>
    <a href='https://github.com/zeng-yifei?tab=repositories/' target='_blank'>Yifei Zeng</a><sup>1</sup>&emsp;
    <a href='https://github.com/YuanxunLu' target='_blank'>Yuanxun Lu</a><sup>1</sup>&emsp;
    <a href='https://github.com/jixinya' target='_blank'>Xinya Ji</a><sup>1</sup>&emsp;
    <a href='https://yoyo000.github.io/' target='_blank'>Yao Yao</a><sup>1</sup>&emsp;
    <a href='https://zhuhao-nju.github.io/home/' target='_blank'>Hao Zhu</a><sup>1+</sup>&emsp;
    <a href='https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html' target='_blank'>Xun Cao</a><sup>1</sup>
</div>
<div>
    <sup>1</sup>Nanjing University
</div>
<div>
    <sup>+</sup>corresponding author
</div>

<h4 align="center">
  <a href="https://zeng-yifei.github.io/avatarbooth_page/" target='_blank'>[Project Page]</a> •
</h4>


# Install

For package installation, ensure that you have installed pytorch (tested on pytorch 2.1 cuda121 and pytorch 1.13 cuda 117):
```bash
pip install -r requirements.txt
```

For data preparation, register and download SMPL models [here](https://smpl.is.tue.mpg.de/). Put the downloaded models in the folder `smpl_models`. The folder structure should look like

```
./
├── ...
└── smpl_models/
    ├── smpl/
        ├── SMPL_FEMALE.pkl
        ├── SMPL_MALE.pkl
        └── SMPL_NEUTRAL.pkl
```

# Usage

To generate avatars. you can use:

```bash
python main.py --mode train --conf confs/examples/obama.conf
```

To use personalized model like LoRA or DreamBooth model, you can assign the corresponding file path in config file like:
```
general {
    sd_path = ... # assign DreamBooth path for whole body in huggingface format, e.g. stabilityai/stable-diffusion-2-1-base or stablediffusionapi/realistic-vision(recommanded and by default)
    sd_face_path = ... # assign DreamBooth path for face in huggingface format

    lora_path = ... # assign lora path with safetensors, e.g. ./pretrained_models/A.safetensors
}
```

To animate the avatar, you can refer to the AvatarCLIP. With the same procedure, you can obtain a animatable fbx after processing the A-pose ply model.


# Teaser
<img src='./assets/exhibit.jpg' height='60%'>

# Animation
<img src='./assets/trump_dances.gif' height='60%'>

## Citation
If you find our work useful for your research, please consider citing the paper:
```
@inproceedings{Zeng2023AvatarBoothHA,
  title={AvatarBooth: High-Quality and Customizable 3D Human Avatar Generation},
  author={Yifei Zeng and Yuanxun Lu and Xinya Ji and Yao Yao and Hao Zhu and Xun Cao},
  year={2023}
}
```

# Acknowledgement
The code is built upon AvatarCLIP and Stable DreamFusion, we express great appreciation to the authors for their great work.