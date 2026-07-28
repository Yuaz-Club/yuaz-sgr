#!/bin/bash
# 每30分钟检查训练状态，异常才关服务器
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin

LOG_FILE="/root/autodl-tmp/logs/train_nohup.out"
STATUS_FILE="/root/autodl-tmp/logs/check_status.txt"
STEP=0

echo "=== $(date) ===" > $STATUS_FILE

# 1. 检查训练进程
PID=$(pgrep -f "train.py" | head -1)
if [ -z "$PID" ]; then
    echo "CRITICAL: 训练进程已死" | tee -a $STATUS_FILE
    # 不要轻易关机, 先输出详细状态
    echo "ps aux | grep python:" >> $STATUS_FILE
    ps aux | grep python >> $STATUS_FILE 2>&1
    echo "最后一次日志:" >> $STATUS_FILE
    tail -5 $LOG_FILE >> $STATUS_FILE 2>&1
    # 只关如果连续两次都死了
    touch /root/autodl-tmp/logs/.train_died_once
    if [ -f /root/autodl-tmp/logs/.train_died_twice ]; then
        echo "两次检查训练都死了，关机" | tee -a $STATUS_FILE
        poweroff
    fi
    mv /root/autodl-tmp/logs/.train_died_once /root/autodl-tmp/logs/.train_died_twice 2>/dev/null
    exit 1
fi
rm -f /root/autodl-tmp/logs/.train_died_once /root/autodl-tmp/logs/.train_died_twice
echo "训练进程: PID=$PID" >> $STATUS_FILE

# 2. 读取最新日志
LAST_LINE=$(tail -1 $LOG_FILE 2>/dev/null)
echo "最后日志: $LAST_LINE" >> $STATUS_FILE

# 3. 提取 feature_loss 检查 NaN
if echo "$LAST_LINE" | grep -q "feature="; then
    FEATURE=$(echo "$LAST_LINE" | grep -oP 'feature=\K[0-9.]+')
    if [ -z "$FEATURE" ] || [ "$FEATURE" = "nan" ] || [ "$FEATURE" = "inf" ]; then
        echo "CRITICAL: feature_loss=$FEATURE, 出现 NaN" | tee -a $STATUS_FILE
        echo "关服务器" | tee -a $STATUS_FILE
        poweroff
        exit 1
    fi
    echo "feature_loss=$FEATURE ✅" >> $STATUS_FILE
fi

if echo "$LAST_LINE" | grep -q "Step"; then
    STEP=$(echo "$LAST_LINE" | grep -oP 'Step \K[0-9]+')
    echo "当前步数: $STEP" >> $STATUS_FILE
fi

# 4. 跑一次快速推理验证
echo "跑推理验证..." >> $STATUS_FILE
cd /root/autodl-tmp/yuaz_sgr
python scripts/quick_test_recon.py >> $STATUS_FILE 2>&1
RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo "推理脚本返回错误码=$RESULT" | tee -a $STATUS_FILE
    # 非致命: 可能是 checkpoint 正在写入中被读到, 记录但不关
fi

echo "=== 检查完成 ✅ ===" >> $STATUS_FILE
cat $STATUS_FILE
