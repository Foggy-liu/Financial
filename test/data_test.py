from data import load_and_parse_data
import pandas as pd
import numpy as np

def load_data(train_path='../train_data.csv', test_path='../test_data.csv'):
    """加载并返回训练集和测试集"""
    print("=" * 60)
    print("1. 加载数据")
    print("=" * 60)

    data = load_and_parse_data(train_path, test_path)

    X_train = data['X_train'].values
    X_test = data['X_test'].values
    y_train = data['y_train']
    y_test = data['y_test']

    # class_dist = pd.Series(y_train).value_counts().sort_index()
    print(f"数据加载完成")
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"标签类别: {np.unique(y_train)}")
    # print(f"类别分布: 0:{class_dist[0]}, 1:{class_dist[1]}, 2:{class_dist[2]}")

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    load_data()
