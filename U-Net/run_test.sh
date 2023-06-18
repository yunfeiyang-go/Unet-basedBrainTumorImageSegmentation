#!/bin/bash
#PBS -S /bin/bash
#PBS -N U-Net
#PBS -q gpuq
#PBS –l nodes=gpu01:ppn=4
#PBS -o test.out                      # 指定标准输出文件
#PBS -e test.err                      # 指定错误输出文件

source /home/apps/anaconda3/etc/profile.d/conda.sh
conda activate torch12
# 执行Python脚本
python /home/ug2020/ug520111910171/bioalgorithm/Brain/U-Net/test.py
python /home/ug2020/ug520111910171/bioalgorithm/Brain/U-Net/test_only_mask.py

# 释放GPU资源
#nvidia-smi --gpu-reset -i 0
