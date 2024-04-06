import re
import os
import pandas as pd
import matplotlib.pyplot as plt

SVM_LOG_PATH = os.path.join(os.getcwd(), 'svm_logs')
MODEL_TEST_LOGS = os.path.join(os.getcwd(), 'model_test_logs')


def extract_metrics(log):
    total_train_acc = re.search(r'Training accuracy: (\d+\.\d+)', log)
    total_test_acc = re.search(r'Testing accuracy: (\d+\.\d+)', log)
    total_time = re.search(r'Total time taken: (\d+\.\d+) seconds', log)

    total_train_acc = float(total_train_acc.group(1)) if total_train_acc else None
    total_test_acc = float(total_test_acc.group(1)) if total_test_acc else None
    total_time = float(total_time.group(1)) if total_time else None

    return total_train_acc, total_test_acc, total_time

def extract_dataset_metrics(log):
    pattern = r'Dataset (\d+)/\d+ - Train Accuracy: (\d+\.\d+), Test Accuracy: (\d+\.\d+)'
    matches = re.findall(pattern, log)
    dataset_metrics = [{'dataset': int(match[0]), 'train_acc': float(match[1]), 'test_acc': float(match[2])} for match in matches]
    return dataset_metrics


def model_test():
    data = []

    for file in os.listdir(MODEL_TEST_LOGS):
        if file.endswith('.log'):
            with open(os.path.join(MODEL_TEST_LOGS, file), 'r') as f:
                log = f.read()
                total_train_acc, total_test_acc, total_time = extract_metrics(log)
                dataset_metrics = extract_dataset_metrics(log)
                model = file.split('.')[0].split('_')[0]
                kernel = file.replace('_result', '').split('.')[0].split('_')[1] if model == 'svm' else 'None'
                data.append({'File': os.path.join(MODEL_TEST_LOGS, file), 'Model': f'{model}_{kernel}', 'Train Accuracy': total_train_acc, 'Test Accuracy': total_test_acc, 'Total Time': total_time})
    
    df = pd.DataFrame(data)
    tasks_train_acc = {f: [] for f in df['Model']}
    tasks_test_acc = {f: [] for f in df['Model']}
    total_time = {f: [] for f in df['Model']}
    svm_kernel_train_acc = {f: [] for f in df['Model'] if f.startswith('svm')}
    svm_kernel_test_acc = {f: [] for f in df['Model'] if f.startswith('svm')}

    for index, row in df.iterrows():
        for task in extract_dataset_metrics(open(row['File'], 'r').read()):
            tasks_test_acc[row['Model']].append(task['test_acc'])
            tasks_train_acc[row['Model']].append(task['train_acc'])
            # 处理 SVM Kernel 的比较
            if row['Model'].startswith('svm'):
                svm_kernel_test_acc[row['Model']].append(task['test_acc'])
                svm_kernel_train_acc[row['Model']].append(task['train_acc'])
        total_time[row['Model']].append(row['Total Time'])

    tasks_test_acc_df = pd.DataFrame(tasks_test_acc, index=range(1, 56))
    tasks_train_acc_df = pd.DataFrame(tasks_train_acc, index=range(1, 56))
    total_time_df = pd.DataFrame(total_time)
    svm_kernel_train_df = pd.DataFrame(svm_kernel_train_acc, index=range(1, 56))
    svm_kernel_test_df = pd.DataFrame(svm_kernel_test_acc, index=range(1, 56))
 
    print("Tasks Test Accuracy DataFrame:")
    print(tasks_test_acc_df)

    print("\nTasks Train Accuracy DataFrame:")
    print(tasks_train_acc_df)

    print("\nTotal Time DataFrame:")
    print(total_time_df)

    print("\nSVM Kernel Train Accuracy DataFrame:")
    print(svm_kernel_train_df)

    print("\nSVM Kernel Test Accuracy DataFrame:")
    print(svm_kernel_test_df)

    tasks_test_acc_df.plot(kind='line', figsize=(15, 8), title='Tasks Test Accuracy')
    plt.xlabel('Dataset No') 
    plt.ylabel('Accuracy')
    plt.savefig('tasks_test_accuracy_plot.png')

    tasks_train_acc_df.plot(kind='line', figsize=(15, 8), title='Tasks Train Accuracy')
    plt.xlabel('Dataset No')
    plt.ylabel('Accuracy')
    plt.savefig('tasks_train_accuracy_plot.png')

    total_time_df.plot(kind='bar', figsize=(15, 8), title='Total Time')
    plt.xlabel('Model') 
    plt.ylabel('Time (s)')
    plt.savefig('total_time_plot.png')

    svm_kernel_train_df.plot(kind='line', figsize=(15, 8), title='SVM Kernel Train Accuracy')
    plt.xlabel('Dataset No') 
    plt.ylabel('Accuracy')
    plt.savefig('model_svm_kernel_train_accuracy_plot.png')

    svm_kernel_test_df.plot(kind='line', figsize=(15, 8), title='SVM Kernel Test Accuracy')
    plt.xlabel('Dataset No')
    plt.ylabel('Accuracy')
    plt.savefig('model_svm_kernel_test_accuracy_plot.png')
    
    plt.show()


def svm_C_test(selected_c_values: list = None):
    data = []

    for file in os.listdir(SVM_LOG_PATH):
        with open(os.path.join(SVM_LOG_PATH, file), 'r') as f:
            log = f.read()
            total_train_acc, total_test_acc, _ = extract_metrics(log)
            model = file.split('.')[0].split('_')[0]
            kernel = file.replace('_result', '').split('.')[0].split('_')[1]
            c = float(file.split('_C')[1].split('.')[0])
            data.append({'Model': f'{model}_{kernel}', 'C': c, 'Train Accuracy': total_train_acc, 'Test Accuracy': total_test_acc})

    df = pd.DataFrame(data)
    if selected_c_values:
        df = df[df['C'].isin(selected_c_values)]
    else:
        selected_c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

    print("SVM C DataFrame:")
    print(df)
    models = df['Model'].unique()
    
    svm_c_train_df = pd.DataFrame(index=selected_c_values, columns=models)
    svm_c_test_df = pd.DataFrame(index=selected_c_values, columns=models)

    for _, row in df.iterrows():
        svm_c_train_df.loc[row['C'], row['Model']] = row['Train Accuracy']
        svm_c_test_df.loc[row['C'], row['Model']] = row['Test Accuracy']

    print("\nSVM C Train Accuracy DataFrame:")
    print(svm_c_train_df)
    print("\nSVM C Test Accuracy DataFrame:")
    print(svm_c_test_df)

    svm_c_train_df.plot(kind='line', figsize=(15, 8), title='SVM Kernel Train Accuracy')
    plt.xlabel('Regularization Coefficient (C)')
    plt.ylabel('Accuracy')
    plt.savefig('svm_c_train_accuracy_plot.png')

    svm_c_test_df.plot(kind='line', figsize=(15, 8), title='SVM Kernel Test Accuracy')
    plt.xlabel('Regularization Coefficient (C)')
    plt.ylabel('Accuracy')
    plt.savefig('svm_c_test_accuracy_plot.png')

    plt.show()


def svm_kernel_test(C_values: list = [1, 2, 3, 4, 5], kernel_functions: list = ['linear', 'poly', 'rbf', 'sigmoid']):
    data = []

    for C in C_values:
        for kernel in kernel_functions:
            file_name = f"svm_{kernel}_C{C}.log"  # 假设日志文件命名规则为 "svm_{kernel}_C{C}.log"
            if os.path.isfile(os.path.join(SVM_LOG_PATH, file_name)):
                with open(os.path.join(SVM_LOG_PATH, file_name), 'r') as f:
                    log = f.read()
                    total_train_acc, total_test_acc, _ = extract_metrics(log)
                    data.append({'C': C, 'Kernel': kernel, 'Train Accuracy': total_train_acc, 'Test Accuracy': total_test_acc})

    df = pd.DataFrame(data)

    print("SVM Kernel DataFrame:")
    print(df)

    svm_kernel_train_df = pd.DataFrame(index=C_values, columns=kernel_functions)
    svm_kernel_test_df = pd.DataFrame(index=C_values, columns=kernel_functions)

    for _, row in df.iterrows():
        svm_kernel_train_df.loc[row['C'], row['Kernel']] = row['Train Accuracy']
        svm_kernel_test_df.loc[row['C'], row['Kernel']] = row['Test Accuracy']

    print("\nSVM Kernel Train Accuracy DataFrame:")
    print(svm_kernel_train_df)
    print("\nSVM Kernel Test Accuracy DataFrame:")
    print(svm_kernel_test_df)

    svm_kernel_train_df.plot(kind='line', figsize=(15, 8), title='SVM Kernel Train Accuracy')
    plt.xlabel('Regularization Coefficient (C)') 
    plt.ylabel('Accuracy') 
    plt.savefig('svm_kernel_train_accuracy_plot.png')

    svm_kernel_test_df.plot(kind='line', figsize=(15, 8), title='SVM Kernel Test Accuracy')
    plt.xlabel('Regularization Coefficient (C)') 
    plt.ylabel('Accuracy') 
    plt.savefig('svm_kernel_test_accuracy_plot.png')

    plt.show()


if __name__ == '__main__':
    model_test()
    svm_C_test(selected_c_values=[i for i in range(11)])
    svm_kernel_test()