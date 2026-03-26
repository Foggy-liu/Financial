import pandas as pd
import numpy as np
import ast
import re


def load_and_parse_data(train_path, test_path):
    """
    加载并解析训练集和测试集数据

    Parameters:
    -----------
    train_path : str
        训练集文件路径
    test_path : str
        测试集文件路径

    Returns:
    --------
    dict: 包含训练集和测试集的特征矩阵和标签
    """

    def parse_vector_pca(df, df_name):
        """
        解析单个数据框的vector_pca列
        """
        print(f"\n{'=' * 50}")
        print(f"处理 {df_name}")
        print(f"{'=' * 50}")

        # 基本信息
        print(f"数据形状：{df.shape}")
        print(f"列名：{list(df.columns)}")
        print(f"标签分布:\n{df['label'].value_counts().sort_index()}")

        # 提取标签
        y = df['label'].values
        print(f"标签数据 shape: {y.shape}")
        print(f"标签类别：{set(y)}")

        # 解析 PCA 向量
        print("\n正在解析 PCA 特征...")

        # 检查第一条数据的格式
        sample = df['vector_pca'].iloc[0]
        print(f"样本数据格式：{type(sample)}")
        print(f"样本内容预览：{str(sample)[:100]}...")

        # 方法1: 通过空格分割解析
        try:
            X_list = []
            for item in df['vector_pca']:
                if isinstance(item, str):
                    # 移除方括号和换行符，按空格分割
                    cleaned = item.replace('[', '').replace(']', '').replace('\n', ' ')
                    # 按空白字符分割并转换为浮点数
                    values = [float(x) for x in cleaned.split() if x.strip()]
                    X_list.append(values)
                else:
                    # 如果已经是列表
                    X_list.append(item)

            X = pd.DataFrame(X_list)
            print(f"✅ 解析成功：通过空格分割")

        except Exception as e:
            print(f"方法1失败，尝试方法2...")

            # 方法2: 使用正则表达式
            try:
                X_list = []
                for item in df['vector_pca']:
                    if isinstance(item, str):
                        # 使用正则表达式提取所有数字
                        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', item)
                        values = [float(n) for n in numbers]
                        X_list.append(values)
                    else:
                        X_list.append(item)

                X = pd.DataFrame(X_list)
                print(f"✅ 解析成功：通过正则表达式")

            except Exception as e2:
                print(f"❌ 解析失败：{str(e2)}")
                raise ValueError(f"无法解析 {df_name} 的 vector_pca 列")

        print(f"PCA 特征数据 shape: {X.shape}")
        print(f"特征维度: {X.shape[1]}")

        # 检查缺失值
        print(f"\n缺失值检查:")
        print(f"X 的缺失值数量：{X.isnull().sum().sum()}")
        print(f"y 的缺失值数量：{np.sum(pd.isnull(y))}")

        return X, y

    # 加载数据
    print("=" * 60)
    print("开始加载数据")
    print("=" * 60)

    # 加载训练集
    train_df = pd.read_csv(train_path)
    X_train, y_train = parse_vector_pca(train_df, "训练集")

    # 加载测试集
    test_df = pd.read_csv(test_path)
    X_test, y_test = parse_vector_pca(test_df, "测试集")

    # 验证数据一致性
    print("\n" + "=" * 50)
    print("数据一致性验证")
    print("=" * 50)
    print(f"训练集特征维度: {X_train.shape[1]}")
    print(f"测试集特征维度: {X_test.shape[1]}")
    print(f"特征维度一致性: {'✓' if X_train.shape[1] == X_test.shape[1] else '✗'}")
    print(f"标签类别一致性: {'✓' if set(y_train) == set(y_test) else '✗'}")

    # 返回数据字典
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_df': train_df,
        'test_df': test_df
    }

    print("\n" + "=" * 50)
    print("数据处理完成！")
    print("=" * 50)
    print(f"训练集: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"测试集: X_test {X_test.shape}, y_test {y_test.shape}")

    return data


# 如果直接运行此脚本，执行数据处理
if __name__ == "__main__":
    # 文件路径
    train_path = "train_data.csv"
    test_path = "test_data.csv"


    # 加载和解析数据
    data = load_and_parse_data(train_path, test_path)


    # 可选：保存处理后的数据为numpy格式，方便后续快速加载
    # np.savez('processed_data.npz',
    #          X_train=data['X_train'].values,
    #          X_test=data['X_test'].values,
    #          y_train=data['y_train'],
    #          y_test=data['y_test'])
    # print("\n✅ 处理后的数据已保存为 'processed_data.npz'")