#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
ls -lh /root/autodl-tmp/checkpoints/model_040000.pt
echo "---size---"
du -sh /root/autodl-tmp/yuaz_sgr/models/
du -sh /root/autodl-tmp/yuaz_sgr/losses/
du -sh /root/autodl-tmp/yuaz_sgr/config/
