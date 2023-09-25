# Simulating Diffusion Bridges with Score Matching

1. 训练代码

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --task gaussian2fourgaussian2d -bt 6000 -it 1000 --M 100 --scheduler cos --noforward  --num_test_samples 500 --drift process  -f
```

2. 加载checkpoint画图（不继续训练）


```bash
CUDA_VISIBLE_DEVICES=1 python main.py --task gaussian2fourgaussian2d -bt 6000 -it 1000 --M 100 --scheduler cos --noforward  --num_test_samples 500 --drift process  -f --checkpoint_score /path/to/checkpoint
```

3. 加载checkpoint继续训练

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --task gaussian2fourgaussian2d -bt 6000 -it 1000 --M 100 --scheduler cos --noforward  --num_test_samples 500 --drift process  -f --checkpoint_score /path/to/checkpoint --continue_score
```