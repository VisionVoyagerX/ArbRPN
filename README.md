# ArbRPN
ArbRPN implemented in pytorch 

Pretrained model is provided

Based on implementation: https://github.com/Lihui-Chen/ArbRPN

Paper link: https://ieeexplore.ieee.org/document/9627886

# Dataset

The GaoFen-2 and WorldView-3 dataset download links can be found in https://github.com/liangjiandeng/PanCollection

# Torch Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ArbRPN                                   [1, 4, 256, 256]          --
├─Conv2d: 1-1                            [1, 64, 256, 256]         640
├─Conv2d: 1-2                            [1, 64, 256, 256]         640
├─Conv2d: 1-3                            [1, 64, 256, 256]         (recursive)
├─Conv2d: 1-4                            [1, 64, 256, 256]         (recursive)
├─Conv2d: 1-5                            [1, 64, 256, 256]         (recursive)
├─ModuleList: 1-14                       --                        (recursive)
│    └─Sequential: 2-1                   [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-1                  [1, 64, 256, 256]         12,352
│    │    └─Sequential: 3-2              [1, 64, 256, 256]         221,568
│    └─Sequential: 2-2                   [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-3                  [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-4              [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-3                   [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-5                  [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-6              [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-4                   [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-7                  [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-8              [1, 64, 256, 256]         (recursive)
├─ModuleList: 1-15                       --                        (recursive)
│    └─Sequential: 2-5                   [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-9                  [1, 64, 256, 256]         12,352
│    │    └─Sequential: 3-10             [1, 64, 256, 256]         221,568
│    └─Sequential: 2-6                   [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-11                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-12             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-7                   [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-13                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-14             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-8                   [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-15                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-16             [1, 64, 256, 256]         (recursive)
├─ModuleList: 1-14                       --                        (recursive)
│    └─Sequential: 2-9                   [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-17                 [1, 64, 256, 256]         12,352
│    │    └─Sequential: 3-18             [1, 64, 256, 256]         221,568
│    └─Sequential: 2-10                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-19                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-20             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-11                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-21                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-22             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-12                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-23                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-24             [1, 64, 256, 256]         (recursive)
├─ModuleList: 1-15                       --                        (recursive)
│    └─Sequential: 2-13                  [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-25                 [1, 64, 256, 256]         12,352
│    │    └─Sequential: 3-26             [1, 64, 256, 256]         221,568
│    └─Sequential: 2-14                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-27                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-28             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-15                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-29                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-30             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-16                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-31                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-32             [1, 64, 256, 256]         (recursive)
├─ModuleList: 1-14                       --                        (recursive)
│    └─Sequential: 2-17                  [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-33                 [1, 64, 256, 256]         12,352
│    │    └─Sequential: 3-34             [1, 64, 256, 256]         221,568
│    └─Sequential: 2-18                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-35                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-36             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-19                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-37                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-38             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-20                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-39                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-40             [1, 64, 256, 256]         (recursive)
├─ModuleList: 1-15                       --                        (recursive)
│    └─Sequential: 2-21                  [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-41                 [1, 64, 256, 256]         12,352
│    │    └─Sequential: 3-42             [1, 64, 256, 256]         221,568
│    └─Sequential: 2-22                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-43                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-44             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-23                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-45                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-46             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-24                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-47                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-48             [1, 64, 256, 256]         (recursive)
├─ModuleList: 1-14                       --                        (recursive)
│    └─Sequential: 2-25                  [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-49                 [1, 64, 256, 256]         12,352
│    │    └─Sequential: 3-50             [1, 64, 256, 256]         221,568
│    └─Sequential: 2-26                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-51                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-52             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-27                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-53                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-54             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-28                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-55                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-56             [1, 64, 256, 256]         (recursive)
├─ModuleList: 1-15                       --                        (recursive)
│    └─Sequential: 2-29                  [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-57                 [1, 64, 256, 256]         12,352
│    │    └─Sequential: 3-58             [1, 64, 256, 256]         221,568
│    └─Sequential: 2-30                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-59                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-60             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-31                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-61                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-62             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-32                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-63                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-64             [1, 64, 256, 256]         (recursive)
├─ModuleList: 1-14                       --                        (recursive)
│    └─Sequential: 2-33                  [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-65                 [1, 64, 256, 256]         12,352
│    │    └─Sequential: 3-66             [1, 64, 256, 256]         221,568
│    └─Sequential: 2-34                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-67                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-68             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-35                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-69                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-70             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-36                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-71                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-72             [1, 64, 256, 256]         (recursive)
├─ModuleList: 1-15                       --                        (recursive)
│    └─Sequential: 2-37                  [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-73                 [1, 64, 256, 256]         12,352
│    │    └─Sequential: 3-74             [1, 64, 256, 256]         221,568
│    └─Sequential: 2-38                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-75                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-76             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-39                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-77                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-78             [1, 64, 256, 256]         (recursive)
│    └─Sequential: 2-40                  [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-79                 [1, 64, 256, 256]         (recursive)
│    │    └─Sequential: 3-80             [1, 64, 256, 256]         (recursive)
├─Conv2d: 1-16                           [1, 1, 256, 256]          577
├─Conv2d: 1-17                           [1, 1, 256, 256]          (recursive)
├─Conv2d: 1-18                           [1, 1, 256, 256]          (recursive)
├─Conv2d: 1-19                           [1, 1, 256, 256]          (recursive)
==========================================================================================
Total params: 2,341,057
Trainable params: 2,341,057
Non-trainable params: 0
Total mult-adds (G): 613.57
==========================================================================================
Input size (MB): 0.33
Forward/backward pass size (MB): 9565.11
Params size (MB): 9.36
Estimated Total Size (MB): 9574.80
==========================================================================================

```

# Quantitative Results

## GaoFen-2

![alt text](https://github.com/nickdndndn/ArbRPN/blob/main/results/Figures_GF2.png?raw=true)

## WorldView-3

![alt text](https://github.com/nickdndndn/ArbRPN/blob/main/results/Figures_WV3.png?raw=true)

# Qualitative Results

## GaoFen-2

![alt text](https://github.com/nickdndndn/ArbRPN/blob/main/results/Images_GF2.png?raw=true)

## WorldView-3

![alt text](https://github.com/nickdndndn/ArbRPN/blob/main/results/Images_WV3.png?raw=true)
