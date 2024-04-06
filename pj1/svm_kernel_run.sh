#!/bin/bash

# 设置参数
model_type="svm"
C=1
max_iter=10000
patience=50
log_dir="svm_kernel_logs"

# 检查日志目录是否存在，不存在则创建
if [ ! -d "$log_dir" ]; then
    mkdir "$log_dir"
fi

# 定义内核数组
kernels=("linear" "poly" "rbf" "sigmoid")

# 循环测试每个内核
for kernel in "${kernels[@]}"; do
    log_file="${log_dir}/svm_${kernel}_C${C}.log"
    command="python main.py --model_type ${model_type} --kernel ${kernel} --C ${C} --max_iter ${max_iter} --patience ${patience} 2>&1 > ${log_file} &"
    echo "Running command: ${command}"
    eval "${command}"
    echo "Started training with kernel: ${kernel}, C: ${C}. Log file: ${log_file}"
done

echo "All experiments started."
