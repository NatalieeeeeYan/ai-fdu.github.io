# Lab1: Report

> 21302010062 宋文彦

## 代码部分

#### 1. 数据集分割：

对cast中的数据，根据其属于测试/训练集进行划分，将正样本赋值为1，负样本为0，存放入 `target_list` 中。同时将对应的diagrams中的特征数据添加到 `data_list` 中。

#### 2. 模型：

调用 sklearn 的包完成了 LR 和 LinearSVM 的训练、测试和评估部分。

由于使用 linear kernel 时，SVM 收敛速度太慢，对 SVM 部分的代码进行了修改。仅对于使用 linear kernel 的情况，设定最大迭代次数 max_iter 和 patience：

- max_iter：可以通过参数 `--max_iter` 设置模型的最大迭代次数，防止因为收敛过慢而一直不停的迭代，程序无法停止。默认值为1000。
- patience：如果模型超过一定次数的迭代后准确率没有提高，就停止迭代。默认值为20。

## 结果

#### 1. 数据处理

由于实验时使用 `nohup ` 重定向输出（便于记录），编写了脚本进行处理，使用 pandas 进行绘图。脚本见：pj1/result_evaluate.py，实验图片见：pj1/*.png。

#### 2. 结果分析

1. 就运行时间而言，使用 Linear kernel 的 SVM 模型的所需时间最长（而且是在未完全收敛的情况下），LR 模型次之，Linear SVM 第三，使用其他核函数的 SVM 模型时间最短。
2. 