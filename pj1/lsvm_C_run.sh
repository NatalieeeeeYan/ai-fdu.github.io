#!/bin/bash

model_type="linear_svm"
log_dir="linear_svm_logs"

if [ ! -d "$log_dir" ]; then
    mkdir "$log_dir"
fi


C_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10 15 20 25 30 40 50 60 70 80 90 100)

# 启动训练，控制变量，挂起后台进程进行测试
for C in "${C_values[@]}"; do
    log_file="${log_dir}/lsvm_${C}.log"
    command="python main.py --model_type ${model_type} --C ${C} 2>&1 > ${log_file} &"
    echo "Running command: ${command}"
    eval "${command}"
    echo "Started training with kernel: ${kernel}, C: ${C}. Log file: ${log_file}"
    sleep 
done

echo "All experiments started."