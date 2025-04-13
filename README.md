# Characters_Detection

# 简介
    A Chinese Characters && Arabic Numbers && Bullseye Target Detection Project Based on YOLO.

# 环境依赖
    1) Request Python3
 
# 目录结构描述
    ├── mainwork.py                     // Main Work written in python
    
    ├── models                      // some {modules}.py && {descript}.yaml

    ├── weights                     // some {weights}.pt
    
    │   ├── arabic_numbers.pt       // YOLOv5-S for arabic numbers detection (from 1 to 8)

    │   ├── characters_2023.pt      // YOLOv5-S for Chinese Number Characters detection (from One to Six)

    │   ├── characters_2024.pt      // YOLOv5-S for Chinese Number Characters detection (from One to Six)

    │   ├── target_shooting_2024.pt // YOLOv5-S for Bullseye Target && AR code detection

    ├── utils                       // some {utilities}.py

    └── ReadMe.md                   // Introduction

# 使用说明
    1) Run mainwork.py after the needed pkgs are all installed.

    2) If u' re using an edge-device like Jetson Xavier NX and wanna sent results to ur robot-controller,
    set param use_usart tobe True plz.
    
    3) [Pretrained 4 models]
        1. Model {arabic_numbers.pt} for 2021年 电赛送药小车项目. It is advisable
    to use a higher confidence threshold, for instance, 0.90.

        2. Model {characters_2023.pt} for 2022 && 2023年 中国机器人及人工智能大赛-无人车视觉巡航项目;

        3. Model {characters_2024.pt} for 2024年 中国机器人及人工智能大赛-无人车视觉巡航项目. It is advisable
    to use a lower confidence threshold, for instance, 0.60.

        4. Model {target_shooting_2024.pt} for 2024年 中国机器人及人工智能大赛-无人车任务挑战赛. Similarly to
    point 3 above, we advise using a lower confidence threshold, such as 0.60.

# 版本内容更新
###### v0(Now):
    1. 基于 YOLOv5 实现检测;
    2. 部署项目到 Windows10 && Jetson AGX;
