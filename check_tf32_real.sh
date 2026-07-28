#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
# 方式1: 独立进程查看默认值
python3 -c "import torch; print('默认:', 'TF32=', torch.backends.cuda.matmul.allow_tf32, 'cudnn=', torch.backends.cudnn.allow_tf32, 'precision=', torch.get_float32_matmul_precision())"

# 方式2: 看训练进程的环境变量
cat /proc/$(pgrep -f "train.py" | head -1)/environ 2>/dev/null | tr '\0' '\n' | grep -i tf32
echo ""

# 方式3: 看训练进程实际写入的日志是否有加速
grep "Step 001\|Step 002\|Step 003" /root/autodl-tmp/logs/train_nohup.out | head -3
