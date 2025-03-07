# Characters_Detection

# 简介
    A Chinese Characters && Arabic Numbers Detection Model Based on YOLO.

# 环境依赖
    1) Request python3
 
# 目录结构描述
    ├── main.py                 // Main Work written in python
    
    ├── models                  // some {modules}.py && {descript}.yaml

    ├── weights                 // some {weights}.pt
    
    │   ├── arabic_numbers.pt   // YOLOv5-S for arabic numbers detection (from 1 to 8)

    │   ├── characters.pt       // YOLOv5-S for Chinese Number Characters detection (from One to Six)

    ├── utils                   // some {utilities}.py

    └── ReadMe.md           // Introduction

# 使用说明
    1) Run mainwork.py after the needed pkgs are all installed.

    2) If u' re using an edge-device like Jetson Xavier NX and wanna sent results to ur robot-controller,
    set param use_usart tobe True plz.
    
    3) [pretrained 2 models]
        1. Model characters.pt for 2022 && 2023年 中国机器人及人工智能大赛-无人车视觉巡航项目;
        2. Model arabic_numbers.pt for 2021年 电赛送药小车项目;

# 版本内容更新
###### v0(Now):
    1. 基于 YOLOv5 实现检测；
    2. 部署项目到 Windows10 && Jetson AGX；
