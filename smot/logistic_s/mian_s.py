import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train_logistic_regression(X_train, y_train, X_test, y_test, cv_folds):
    """
    使用指定折数进行交叉验证训练逻辑回归模型
    """
    print("\n" + "=" * 70)
    print(f"逻辑回归训练 - {cv_folds}折交叉验证")
    print("=" * 70)

    # 参数设置
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    solvers = ['lbfgs', 'newton-cg', 'sag', 'liblinear']

    results = []

    print("\n开始网格搜索...")
    print(f"参数组合: {len(C_values)}个C值 × {len(solvers)}个求解器 × 2种策略 = {len(C_values) * len(solvers) * 2}种")
    print("-" * 50)

    for solver in solvers:
        for multi_class in ['ovr', 'multinomial']:
            # liblinear不支持multinomial
            if solver == 'liblinear' and multi_class == 'multinomial':
                continue

            print(f"\n测试: solver={solver}, multi_class={multi_class}")

            for C in C_values:
                model = LogisticRegression(
                    C=C,
                    solver=solver,
                    multi_class=multi_class,
                    max_iter=1000,
                    random_state=42
                )

                # 交叉验证
                auc_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc_ovr')
                f1_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_macro')
                accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

                mean_auc = np.mean(auc_scores)
                mean_f1 = np.mean(f1_scores)
                mean_accuracy = np.mean(accuracy_scores)

                print(f"  C={C}: AUC={mean_auc:.4f}, F1={mean_f1:.4f}, Accuracy={mean_accuracy:.4f}")

                results.append({
                    'C': C,
                    'solver': solver,
                    'multi_class': multi_class,
                    'auc': mean_auc,
                    'f1': mean_f1,
                    'accuracy': mean_accuracy
                })

    # 转换为DataFrame并找出最佳参数（按AUC排序）
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('auc', ascending=False)

    best = results_df.iloc[0]

    print("\n" + "=" * 50)
    print(f"{cv_folds}折交叉验证 - 最佳参数:")
    print(f"  C = {best['C']}")
    print(f"  solver = {best['solver']}")
    print(f"  multi_class = {best['multi_class']}")
    print(f"  交叉验证AUC = {best['auc']:.4f}")
    print(f"  交叉验证F1 = {best['f1']:.4f}")
    print(f"  交叉验证准确率 = {best['accuracy']:.4f}")
    print("=" * 50)

    # 用最佳参数训练最终模型
    best_params = {
        'C': best['C'],
        'solver': best['solver'],
        'multi_class': best['multi_class']
    }

    final_model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
    final_model.fit(X_train, y_train)

    # 测试集评估
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    test_f1_macro = f1_score(y_test, y_pred, average='macro')
    test_f1_per_class = f1_score(y_test, y_pred, average=None)

    print(f"\n{cv_folds}折交叉验证 - 测试集结果:")
    print(f"  准确率: {test_accuracy:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  F1宏观平均: {test_f1_macro:.4f}")

    print("\n各类别F1:")
    target_names = ['负面(0)', '中性(1)', '正面(2)']
    for i, label in enumerate(target_names):
        print(f"  {label}: {test_f1_per_class[i]:.4f}")

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return final_model, results_df, test_auc, test_f1_macro, test_accuracy


def main():
    """主函数"""
    print("=" * 70)
    print("逻辑回归模型训练（SMOTE过采样后）")
    print("=" * 70)

    # 1. 加载数据
    print("\n加载数据...")
    train_data = np.load('../train_data_smote.npz')
    X_train = train_data['X_train']
    y_train = train_data['y_train']

    test_data = np.load('../test_data.npz')
    X_test = test_data['X_test']
    y_test = test_data['y_test']

    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")

    # 2. 5折交叉验证
    model_5, results_5, auc_5, f1_5, acc_5 = train_logistic_regression(
        X_train, y_train, X_test, y_test, cv_folds=5
    )

    # 3. 10折交叉验证
    model_10, results_10, auc_10, f1_10, acc_10 = train_logistic_regression(
        X_train, y_train, X_test, y_test, cv_folds=10
    )

    # 4. 对比结果
    print("5折 vs 10折交叉验证对比")
    print("=" * 70)

    comparison = pd.DataFrame({
        '折数': ['5折', '10折'],
        '最佳参数': [
            f"C={results_5.iloc[0]['C']}, solver={results_5.iloc[0]['solver']}, multi_class={results_5.iloc[0]['multi_class']}",
            f"C={results_10.iloc[0]['C']}, solver={results_10.iloc[0]['solver']}, multi_class={results_10.iloc[0]['multi_class']}"
        ],
        '交叉验证AUC': [results_5.iloc[0]['auc'], results_10.iloc[0]['auc']],
        '测试集AUC': [auc_5, auc_10],
        '测试集F1': [f1_5, f1_10],
        '测试集准确率': [acc_5, acc_10]
    })

    print(comparison.to_string(index=False))

    # 5. 选择表现更好的模型
    if auc_5 >= auc_10:
        best_model = model_5
        best_folds = 5
        best_auc = auc_5
        best_f1 = f1_5
        best_acc = acc_5
        print(f"\n选择 {best_folds}折交叉验证的模型（测试集AUC={best_auc:.4f}）")
    else:
        best_model = model_10
        best_folds = 10
        best_auc = auc_10
        best_f1 = f1_10
        best_acc = acc_10
        print(f"\n选择 {best_folds}折交叉验证的模型（测试集AUC={best_auc:.4f}）")

    # 6. 绘制最终模型的混淆矩阵
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面(0)', '中性(1)', '正面(2)'],
                yticklabels=['负面(0)', '中性(1)', '正面(2)'],
                annot_kws={'size': 14})

    plt.title(f'混淆矩阵 - 逻辑回归 (SMOTE后, {best_folds}折交叉验证)', fontsize=14)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_final.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n混淆矩阵已保存为 'confusion_matrix_final.png'")

    # 7. 保存最佳模型
    # joblib.dump(best_model, 'best_logistic_model_smote.pkl')
    # print(f"最佳模型已保存为 'best_logistic_model_smote.pkl'")

    # 8. 保存对比结果
    # comparison.to_csv('cv_comparison_results.csv', index=False)
    # print("对比结果已保存为 'cv_comparison_results.csv'")

    print("实验完成！")



if __name__ == "__main__":
    main()