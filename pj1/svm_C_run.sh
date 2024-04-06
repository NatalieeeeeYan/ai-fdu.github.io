#!/bin/bash

model_type="svm"
max_iter=10000
patience=50
log_dir="svm_logs"

if [ ! -d "$log_dir" ]; then
    mkdir "$log_dir"
fi

# 定义内核和 C 值数组
kernels=("linear" "poly" "rbf" "sigmoid")
# kernels=("rbf")       # 用于查漏补缺

C_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 15 20 25 30 40 50 60 70 80 90 100)
C_values=(40 50 60 70 80 90)        # 用于查漏补缺

# 启动训练，控制变量，挂起后台进程进行测试
for kernel in "${kernels[@]}"; do
    for C in "${C_values[@]}"; do
        log_file="${log_dir}/svm_${kernel}_C${C}.log"
        command="python main.py --model_type ${model_type} --kernel ${kernel} --C ${C} --max_iter ${max_iter} --patience ${patience} 2>&1 > ${log_file} &"
        echo "Running command: ${command}"
        eval "${command}"
        echo "Started training with kernel: ${kernel}, C: ${C}. Log file: ${log_file}"
        sleep 
    done
done

echo "All experiments started."