import os
import argparse
import numpy as np
import pandas as pd
import json
import time

from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC, LinearSVC
from fea import feature_extraction
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from Bio.PDB import PDBParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold


class SVMModel:
    def __init__(self, kernel='rbf', C=1.0, max_iter=1000, patience=10):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.patience = patience
        if kernel == 'linear':
            self.model = SVC(kernel=kernel, C=C, max_iter=max_iter)
        else:
            self.model = SVC(kernel=kernel, C=C)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)
        

    def evaluate(self, data, targets):
        return self.model.score(data, targets)

class LRModel:
    # todo: 
    def __init__(self, C: float = 1.0) -> None:
        """
            Initialize Logistic Regression (from sklearn) model.

            Parameters:
            - C (float): Inverse of regularization strength; must be a positive float. Default is 1.0.
        """
        self.C = C
        self.model = LogisticRegression(C=C)

    def train(self, train_data, train_targets):
        """
            Train the Logistic Regression model.

            Parameters:
            - train_data (array-like): Training data.
            - train_targets (array-like): Target values for the training data.
        """
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        """
            Evaluate the performance of the Logistic Regression model.

            Parameters:
            - data (array-like): Data to be evaluated.
            - targets (array-like): True target values corresponding to the data.

            Returns:
            - float: Accuracy score of the model on the given data.
        """
        return self.model.score(data, targets)

class LinearSVMModel:
    # todo
    def __init__(self, C: float=1.0) -> None:
        """
            Initialize Linear SVM (from sklearn) model.

            Parameters:
            - C (float): Inverse of regularization strength; must be a positive float. Default is 1.0.
        """
        self.C = C
        self.model = LinearSVC(C=C)

    """
        Train and Evaluate are the same.
    """

    def train(self, train_data, train_targets):
        """
            Train the Linear SVM model.

            Parameters:
            - train_data (array-like): Training data.
            - train_targets (array-like): Target values for the training data.
        """
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        """
            Evaluate the performance of the Linear SVM model.

            Parameters:
            - data (array-like): Data to be evaluated.
            - targets (array-like): True target values corresponding to the data.

            Returns:
            - float: Accuracy score of the model on the given data.
        """
        return self.model.score(data, targets)


def data_preprocess(args):
    if args.ent:
        diagrams = feature_extraction()[0]
    else:
        diagrams = np.load('./data/diagrams.npy')
    
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []

    if args.KFold:
        kf = KFold(n_splits=args.KFold, shuffle=True, random_state=42)  # 使用指定折数的交叉验证
    else:
        kf = None

    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task]
        # todo: Try to load data/target
        
        # 使用交叉验证
        if kf:
            for train_indices, test_indices in kf.split(diagrams):
                train_data = diagrams[train_indices].tolist()
                train_targets = task_col.iloc[train_indices].apply(lambda x: 1 if x in [1, 3] else 0).tolist()

                test_data = diagrams[test_indices].tolist()
                test_targets = task_col.iloc[test_indices].apply(lambda x: 1 if x in [1, 3] else 0).tolist()

                if args.random_forest:
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(train_data, train_targets)
                    selected_features = rf.feature_importances_ > 0.01
                    train_data = np.array(train_data)[:, selected_features]
                    test_data = np.array(test_data)[:, selected_features]
                
                elif args.tsne:
                    tsne = TSNE(n_components=2)
                    train_data = np.array(train_data)
                    train_data = tsne.fit_transform(train_data)
                    test_data = np.array(test_data)
                    test_data = tsne.fit_transform(test_data)

                elif args.pca:
                    pca = PCA(n_components=100)
                    train_data = pca.fit_transform(train_data)
                    test_data = pca.transform(test_data)
                
                elif args.kernel_pca:
                    kpca = KernelPCA(n_components=100, kernel='rbf')
                    train_data = kpca.fit_transform(train_data)
                    test_data = kpca.transform(test_data)

                data_list.append((train_data, test_data)) 
                target_list.append((train_targets, test_targets))
        
        else:
            train_indices = np.where((task_col == 1) | (task_col == 2))[0]
            test_indices = np.where((task_col == 3) | (task_col == 4))[0]
            
            train_data = diagrams[train_indices].tolist()
            train_targets = task_col.iloc[train_indices].apply(lambda x: 1 if x in [1, 3] else 0).tolist()

            test_data = diagrams[test_indices].tolist()
            test_targets = task_col.iloc[test_indices].apply(lambda x: 1 if x in [1, 3] else 0).tolist()

            if args.random_forest:
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(train_data, train_targets)
                selected_features = rf.feature_importances_ > 0.01
                train_data = np.array(train_data)[:, selected_features]
                test_data = np.array(test_data)[:, selected_features]
            
            elif args.tsne:
                tsne = TSNE(n_components=2)
                train_data = np.array(train_data)
                train_data = tsne.fit_transform(train_data)
                test_data = np.array(test_data)
                test_data = tsne.fit_transform(test_data)
        
            elif args.pca:
                pca = PCA(n_components=100)
                train_data = pca.fit_transform(train_data)
                test_data = pca.transform(test_data)
            
            elif args.kernel_pca:
                kpca = KernelPCA(n_components=100, kernel='rbf')
                train_data = kpca.fit_transform(train_data)
                test_data = kpca.transform(test_data)

            data_list.append((train_data, test_data)) 
            target_list.append((train_targets, test_targets))

    return data_list, target_list


def main(args):
    start_total_time = time.time()

    data_list, target_list = data_preprocess(args)

    task_acc_train = []
    task_acc_test = []

    # Model Initialization based on input argument
    if args.model_type == 'svm':
        model = SVMModel(kernel=args.kernel, C=args.C)
    else:
        print("Attention: Kernel option is not supported")
        if args.model_type == 'linear_svm':
            model = LinearSVMModel(C=args.C)
        elif args.model_type == 'lr':
            model = LRModel(C=args.C)
        else:
            raise ValueError("Unsupported model type")

    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i+1}/{len(data_list)}")

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)


    end_total_time = time.time()
    
    print("Training accuracy:", sum(task_acc_train)/len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test)/len(task_acc_test))
    print(f"Total time taken: {end_total_time - start_total_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Model Training and Evaluation")
    parser.add_argument('--model_type', type=str, default='svm', choices=['svm', 'linear_svm', 'lr'], help="Model type")
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'], help="Kernel type")
    parser.add_argument('--C', type=float, default=20, help="Regularization parameter")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations for linear SVM")
    parser.add_argument('--patience', type=int, default=20, help="Number of epochs to wait for improvement for linear SVM")
    parser.add_argument('--KFold', type=int, default=0, help="Number of folds for KFold cross validation")

    parser.add_argument('--pca', type=int, default=0, help='Use PCA to decomposite data or not. ')
    parser.add_argument('--kernel_pca', type=int, default=0, help='Use Kernel PCA to decomposite data or not. ')
    parser.add_argument('--tsne', type=int, default=0, help='Use t-SNE to decomposite data or not. ')
    parser.add_argument('--random_forest', type=int, default=0, help='Use Random Forest to select features or not. ')

    parser.add_argument('--ent', action='store_true', help="Load data from a file using a feature engineering function feature_extraction() from fea.py")
    args = parser.parse_args()
    main(args)