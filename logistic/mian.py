import matplotlib

matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, log_loss, f1_score
import joblib
import warnings

warnings.filterwarnings('ignore')

from data import load_and_parse_data

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


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

    # 显示类别分布
    class_dist = pd.Series(y_train).value_counts().sort_index()
    print(f"数据加载完成")
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"标签类别: {np.unique(y_train)}")
    print(f"类别分布: 0:{class_dist[0]}, 1:{class_dist[1]}, 2:{class_dist[2]}")

    return X_train, X_test, y_train, y_test


def tune_hyperparameters(X_train, y_train, X_test, y_test):
    """超参数调优并绘制调优过程"""
    print("\n" + "=" * 60)
    print("2. 超参数调优")
    print("=" * 60)

    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    solvers = ['lbfgs', 'newton-cg', 'sag', 'liblinear']

    results_df = pd.DataFrame()

    print("测试不同参数组合...")

    for solver in solvers:
        for multi_class in ['ovr', 'multinomial']:
            if solver == 'liblinear' and multi_class == 'multinomial':
                continue

            scores_auc = []
            scores_recall = []
            scores_accuracy = []
            scores_cross_entropy = []
            scores_f1 = []  # 新增F1分数

            for C in C_values:
                model = LogisticRegression(
                    C=C,
                    solver=solver,
                    multi_class=multi_class,
                    max_iter=1000,
                    random_state=42
                )

                auc_scores = cross_val_score(model, X_train, y_train, cv=5,
                                             scoring='roc_auc_ovr')
                recall_scores = cross_val_score(model, X_train, y_train, cv=5,
                                                scoring='recall_macro')
                accuracy_scores = cross_val_score(model, X_train, y_train, cv=5,
                                                  scoring='accuracy')
                f1_scores = cross_val_score(model, X_train, y_train, cv=5,
                                            scoring='f1_macro')  # 新增F1交叉验证

                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)
                ce_score = log_loss(y_test, y_pred_proba)

                scores_auc.append(np.mean(auc_scores))
                scores_recall.append(np.mean(recall_scores))
                scores_accuracy.append(np.mean(accuracy_scores))
                scores_cross_entropy.append(ce_score)
                scores_f1.append(np.mean(f1_scores))  # 新增

            temp_df = pd.DataFrame({
                'C': C_values,
                'AUC': scores_auc,
                'Recall': scores_recall,
                'Accuracy': scores_accuracy,
                'F1': scores_f1,  # 新增
                'CrossEntropy': scores_cross_entropy,
                'Solver': solver,
                'MultiClass': multi_class
            })
            results_df = pd.concat([results_df, temp_df], ignore_index=True)

    # 绘制调优过程（增加F1子图）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 改为2行3列

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    line_styles = ['-', '--', '-.', ':']

    metrics = ['AUC', 'Recall', 'Accuracy', 'F1', 'CrossEntropy']  # 增加F1
    titles = ['AUC分数变化', '召回率变化', '准确率变化', 'F1分数变化', '交叉熵损失变化']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]

        param_combinations = [
            ('lbfgs', 'ovr'), ('lbfgs', 'multinomial'),
            ('newton-cg', 'ovr'), ('newton-cg', 'multinomial'),
            ('sag', 'ovr'), ('sag', 'multinomial'),
            ('liblinear', 'ovr')
        ]

        for i, (solver, multi_class) in enumerate(param_combinations):
            subset = results_df[(results_df['Solver'] == solver) &
                                (results_df['MultiClass'] == multi_class)]

            if not subset.empty:
                label = f"{solver}-{multi_class}"
                ax.plot(subset['C'], subset[metric],
                        marker='o', linestyle=line_styles[i % 4],
                        color=colors[i % 6], label=label, linewidth=2, markersize=6)

        ax.set_xscale('log')
        ax.set_xlabel('C值')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle('逻辑回归超参数调优过程（含F1分数）', fontsize=16)
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    print("调优过程图已保存为 'hyperparameter_tuning.png'")

    try:
        plt.show()
    except:
        print("图形显示失败，但已保存图片")
        plt.close()

    # 找出各指标最佳参数（新增F1最佳）
    best_auc = results_df.loc[results_df['AUC'].idxmax()]
    best_recall = results_df.loc[results_df['Recall'].idxmax()]
    best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['F1'].idxmax()]  # 新增
    best_ce = results_df.loc[results_df['CrossEntropy'].idxmin()]

    print("\n各指标最佳参数:")
    print(f"AUC最佳: C={best_auc['C']}, solver={best_auc['Solver']}, "
          f"multi_class={best_auc['MultiClass']}, AUC={best_auc['AUC']:.4f}")
    print(f"召回率最佳: C={best_recall['C']}, solver={best_recall['Solver']}, "
          f"multi_class={best_recall['MultiClass']}, Recall={best_recall['Recall']:.4f}")
    print(f"准确率最佳: C={best_accuracy['C']}, solver={best_accuracy['Solver']}, "
          f"multi_class={best_accuracy['MultiClass']}, Accuracy={best_accuracy['Accuracy']:.4f}")
    print(f"F1最佳: C={best_f1['C']}, solver={best_f1['Solver']}, "
          f"multi_class={best_f1['MultiClass']}, F1={best_f1['F1']:.4f} ⭐")  # 新增
    print(f"交叉熵最佳: C={best_ce['C']}, solver={best_ce['Solver']}, "
          f"multi_class={best_ce['MultiClass']}, CrossEntropy={best_ce['CrossEntropy']:.4f}")

    # 用F1作为主要指标选择最佳参数
    best_params = {
        'C': best_f1['C'],
        'solver': best_f1['Solver'],
        'multi_class': best_f1['MultiClass']
    }

    return best_params, results_df


def train_final_model(X_train, y_train, X_test, y_test, best_params):
    """使用最佳参数训练最终模型（对比有无类别平衡）"""
    print("\n" + "=" * 60)
    print("3. 最终模型训练与评估")
    print("=" * 60)

    target_names = ['负面(0)', '中性(1)', '正面(2)']

    # 训练两个模型做对比：无权重 vs 平衡权重
    models = {
        '无权重': LogisticRegression(**best_params, max_iter=1000, random_state=42),
        '平衡权重': LogisticRegression(**best_params, class_weight='balanced', max_iter=1000, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- {name}模型 ---")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        ce = log_loss(y_test, y_pred_proba)

        # 计算F1分数
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        print(f"准确率: {accuracy:.4f}")
        print(f"AUC分数: {auc:.4f}")
        print(f"交叉熵损失: {ce:.4f}")
        print(f"F1宏观平均: {f1_macro:.4f} ⭐")
        print(f"F1加权平均: {f1_weighted:.4f}")
        print("\n各类别F1:")
        for i, label in enumerate(target_names):
            print(f"  {label}: {f1_per_class[i]:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'ce': ce,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'y_pred': y_pred
        }

    # 用F1宏观平均选择最佳模型
    if results['平衡权重']['f1_macro'] > results['无权重']['f1_macro']:
        print("\n✅ 平衡权重模型表现更好（F1宏观平均更高），选择此模型")
        final_model = results['平衡权重']['model']
        final_accuracy = results['平衡权重']['accuracy']
        final_auc = results['平衡权重']['auc']
        final_ce = results['平衡权重']['ce']
        final_f1_macro = results['平衡权重']['f1_macro']
        final_f1_weighted = results['平衡权重']['f1_weighted']
        final_f1_per_class = results['平衡权重']['f1_per_class']
        y_pred = results['平衡权重']['y_pred']
        weight_used = 'balanced'
    else:
        print("\n⚠️ 无权重模型表现更好（F1宏观平均更高），选择此模型")
        final_model = results['无权重']['model']
        final_accuracy = results['无权重']['accuracy']
        final_auc = results['无权重']['auc']
        final_ce = results['无权重']['ce']
        final_f1_macro = results['无权重']['f1_macro']
        final_f1_weighted = results['无权重']['f1_weighted']
        final_f1_per_class = results['无权重']['f1_per_class']
        y_pred = results['无权重']['y_pred']
        weight_used = None

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    title = '混淆矩阵 - 最终模型'
    if weight_used:
        title += ' (带类别平衡)'
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png', dpi=300)

    try:
        plt.show()
    except:
        print("混淆矩阵显示失败，但已保存图片")
        plt.close()

    return final_model, final_accuracy, final_auc, final_ce, final_f1_macro, final_f1_weighted, final_f1_per_class, weight_used


def main():
    """主函数"""
    print("=" * 70)
    print("逻辑回归模型训练与超参数调优")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_data()

    best_params, tuning_results = tune_hyperparameters(X_train, y_train, X_test, y_test)

    final_model, accuracy, auc, ce, f1_macro, f1_weighted, f1_per_class, weight_used = train_final_model(
        X_train, y_train, X_test, y_test, best_params
    )

    print("\n" + "=" * 60)
    print("4. 保存结果")
    print("=" * 60)

    model_info = {
        'model': final_model,
        'best_params': {**best_params, 'class_weight': weight_used},
        'accuracy': accuracy,
        'auc': auc,
        'cross_entropy': ce,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class.tolist() if f1_per_class is not None else None
    }
    joblib.dump(model_info, 'best_logistic_model.pkl')
    print("模型已保存为 'best_logistic_model.pkl'")

    tuning_results.to_csv('tuning_results.csv', index=False)
    print("调优结果已保存为 'tuning_results.csv'")

    print("\n" + "=" * 70)
    print("训练完成！最终结果总结")
    print("=" * 70)
    print(f"最佳参数: C={best_params['C']}, solver={best_params['solver']}, "
          f"multi_class={best_params['multi_class']}")
    print(f"类别平衡: {'使用平衡权重' if weight_used else '未使用'}")
    print(f"\n【核心指标 - F1宏观平均】: {f1_macro:.4f}")
    print("\n【各类别F1】:")
    target_names = ['负面(0)', '中性(1)', '正面(2)']
    for i, label in enumerate(target_names):
        print(f"  {label}: {f1_per_class[i]:.4f}")
    print(f"\n【其他指标】:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  AUC分数: {auc:.4f}")
    print(f"  交叉熵损失: {ce:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()