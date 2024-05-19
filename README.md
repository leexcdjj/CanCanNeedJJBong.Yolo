# Yolo.Net

本项目集成封装了Yolo各种推理模型，包括且不限于Cls、Seg、Obb、Pose

支持Cpu、Gpu，免Cuda

**注意: 目前只支持YoloV8**

## 效果

cls

![2cls.jpg](assets/螺丝刀2cls.jpg)

Detection

![OBB2Detection.jpg](assets/OBB测试飞机场2Detection.jpg)

Seg

![83c6a5691f1464f9c0e963f5d42bf3e0Seg.jpg](assets/83c6a5691f1464f9c0e963f5d42bf3e0Seg.jpg)

Pose

![4KeyPoints.jpg](assets/4KeyPoints.jpg)

OBB

![OBBOBB.jpg](assets/OBB测试球场OBB.jpg)

## 安装

```
dotnet add package CanCanNeedJJBong.Yolo --version 1.0.0
```

## 用法

封装YoloService，参考TestConsole项目
