import matplotlib

matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
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

    class_dist = pd.Series(y_train).value_counts().sort_index()
    print(f"数据加载完成")
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"标签类别: {np.unique(y_train)}")
    print(f"类别分布: 0:{class_dist[0]}, 1:{class_dist[1]}, 2:{class_dist[2]}")

    return X_train, X_test, y_train, y_test


def tune_xgboost_hyperparameters(X_train, y_train, X_test, y_test):
    """XGBoost超参数调优（选择三个最重要参数）"""
    print("\n" + "=" * 60)
    print("2. XGBoost超参数调优")
    print("=" * 60)

    # XGBoost三个最重要的参数
    n_estimators_list = [50, 100, 200]  # 树的数量
    max_depth_list = [3, 6, 9]  # 树的最大深度（XGBoost默认6）
    learning_rate_list = [0.01, 0.1, 0.3]  # 学习率

    total_combinations = len(n_estimators_list) * len(max_depth_list) * len(learning_rate_list)
    print(f"测试参数组合... 共 {total_combinations} 种组合")

    results_df = pd.DataFrame()
    combination_count = 0

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for learning_rate in learning_rate_list:
                combination_count += 1
                if combination_count % 5 == 0:
                    print(f"  已测试 {combination_count}/{total_combinations} 个组合")

                model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    objective='multi:softprob',  # 多分类
                    num_class=3,  # 3个类别
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='mlogloss'
                )

                # 交叉验证
                auc_scores = cross_val_score(model, X_train, y_train, cv=5,
                                             scoring='roc_auc_ovr')
                accuracy_scores = cross_val_score(model, X_train, y_train, cv=5,
                                                  scoring='accuracy')
                f1_scores = cross_val_score(model, X_train, y_train, cv=5,
                                            scoring='f1_macro')

                # 训练模型计算测试集交叉熵
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)
                ce_score = log_loss(y_test, y_pred_proba)

                temp_df = pd.DataFrame({
                    'n_estimators': [n_estimators],
                    'max_depth': [max_depth],
                    'learning_rate': [learning_rate],
                    'AUC': [np.mean(auc_scores)],
                    'Accuracy': [np.mean(accuracy_scores)],
                    'F1': [np.mean(f1_scores)],
                    'CrossEntropy': [ce_score]
                })
                results_df = pd.concat([results_df, temp_df], ignore_index=True)

    print("参数测试完成！")

    # 绘制调优过程（4个指标）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1：n_estimators的影响
    ax1 = axes[0, 0]
    for max_depth in max_depth_list:
        subset = results_df[results_df['max_depth'] == max_depth]
        if not subset.empty:
            avg_f1 = subset.groupby('n_estimators')['F1'].mean()
            ax1.plot(avg_f1.index, avg_f1.values, 'o-', label=f'max_depth={max_depth}')
    ax1.set_xlabel('n_estimators (树的数量)')
    ax1.set_ylabel('F1分数')
    ax1.set_title('树数量对F1分数的影响')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2：max_depth的影响
    ax2 = axes[0, 1]
    for learning_rate in learning_rate_list:
        subset = results_df[results_df['learning_rate'] == learning_rate]
        if not subset.empty:
            avg_f1 = subset.groupby('max_depth')['F1'].mean()
            ax2.plot(avg_f1.index, avg_f1.values, 'o-', label=f'learning_rate={learning_rate}')
    ax2.set_xlabel('max_depth (树的最大深度)')
    ax2.set_ylabel('F1分数')
    ax2.set_title('最大深度对F1分数的影响')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3：learning_rate的影响
    ax3 = axes[1, 0]
    for n_estimators in n_estimators_list:
        subset = results_df[results_df['n_estimators'] == n_estimators]
        if not subset.empty:
            avg_f1 = subset.groupby('learning_rate')['F1'].mean()
            ax3.plot(avg_f1.index, avg_f1.values, 'o-', label=f'n_estimators={n_estimators}')
    ax3.set_xlabel('learning_rate (学习率)')
    ax3.set_ylabel('F1分数')
    ax3.set_title('学习率对F1分数的影响')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4：AUC变化
    ax4 = axes[1, 1]
    for learning_rate in learning_rate_list:
        subset = results_df[results_df['learning_rate'] == learning_rate]
        if not subset.empty:
            avg_auc = subset.groupby('max_depth')['AUC'].mean()
            ax4.plot(avg_auc.index, avg_auc.values, 'o-', label=f'learning_rate={learning_rate}')
    ax4.set_xlabel('max_depth (树的最大深度)')
    ax4.set_ylabel('AUC分数')
    ax4.set_title('AUC分数变化（对比）')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('XGBoost超参数调优过程', fontsize=16)
    plt.tight_layout()
    plt.savefig('xgboost_tuning.png', dpi=300, bbox_inches='tight')
    print("调优过程图已保存为 'xgboost_tuning.png'")

    try:
        plt.show()
    except:
        print("图形显示失败，但已保存图片")
        plt.close()

    # 找出最佳参数（用F1作为主要指标）
    best_idx = results_df['F1'].idxmax()
    best_params_row = results_df.loc[best_idx]

    print("\n" + "=" * 60)
    print("最佳参数组合 (基于F1分数):")
    print(f"  n_estimators: {int(best_params_row['n_estimators'])}")
    print(f"  max_depth: {int(best_params_row['max_depth'])}")
    print(f"  learning_rate: {best_params_row['learning_rate']}")
    print(f"  F1: {best_params_row['F1']:.4f} ⭐")
    print(f"  AUC: {best_params_row['AUC']:.4f}")
    print(f"  Accuracy: {best_params_row['Accuracy']:.4f}")

    # 显示Top 3最佳组合
    print("\nTop 3 最佳参数组合 (按F1排序):")
    top3 = results_df.nlargest(3, 'F1')[['n_estimators', 'max_depth', 'learning_rate', 'F1', 'AUC']]
    print(top3.to_string(index=False))

    best_params = {
        'n_estimators': int(best_params_row['n_estimators']),
        'max_depth': int(best_params_row['max_depth']),
        'learning_rate': best_params_row['learning_rate']
    }

    return best_params, results_df


def train_final_model(X_train, y_train, X_test, y_test, best_params):
    """使用最佳参数训练最终模型（对比有无类别平衡）"""
    print("\n" + "=" * 60)
    print("3. 最终模型训练与评估")
    print("=" * 60)

    target_names = ['负面(0)', '中性(1)', '正面(2)']

    # XGBoost的类别平衡参数是scale_pos_weight，但对多分类需要用sample_weight
    # 这里用两种方式对比
    models = {
        '无权重': XGBClassifier(**best_params, objective='multi:softprob', num_class=3,
                                random_state=42, n_jobs=-1, eval_metric='mlogloss'),
        '平衡权重': XGBClassifier(**best_params, objective='multi:softprob', num_class=3,
                                  random_state=42, n_jobs=-1, eval_metric='mlogloss')
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- {name}模型 ---")

        if name == '平衡权重':
            # 计算样本权重（类别平衡）
            class_dist = pd.Series(y_train).value_counts()
            n_samples = len(y_train)
            n_classes = len(class_dist)
            sample_weights = np.zeros_like(y_train, dtype=float)
            for cls in range(n_classes):
                mask = (y_train == cls)
                weight = n_samples / (n_classes * class_dist[cls])
                sample_weights[mask] = weight
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
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
    title = '混淆矩阵 - XGBoost最终模型'
    if weight_used:
        title += ' (带类别平衡)'
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('xgboost_confusion_matrix.png', dpi=300)

    try:
        plt.show()
    except:
        print("混淆矩阵显示失败，但已保存图片")
        plt.close()

    # 特征重要性分析
    plt.figure(figsize=(10, 6))
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    plt.bar(range(20), importances[indices])
    plt.xticks(range(20), [f'PC{i + 1}' for i in indices], rotation=45)
    plt.xlabel('PCA特征')
    plt.ylabel('特征重要性')
    plt.title('XGBoost特征重要性排名 (Top 20)')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=300)

    try:
        plt.show()
    except:
        print("特征重要性图显示失败，但已保存图片")
        plt.close()

    return final_model, final_accuracy, final_auc, final_ce, final_f1_macro, final_f1_weighted, final_f1_per_class, weight_used


def main():
    """主函数"""
    print("=" * 70)
    print("XGBoost模型训练与超参数调优")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_data()

    best_params, tuning_results = tune_xgboost_hyperparameters(X_train, y_train, X_test, y_test)

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
    joblib.dump(model_info, 'best_xgboost_model.pkl')
    print("模型已保存为 'best_xgboost_model.pkl'")

    tuning_results.to_csv('xgboost_tuning_results.csv', index=False)
    print("调优结果已保存为 'xgboost_tuning_results.csv'")

    print("\n" + "=" * 70)
    print("训练完成！最终结果总结")
    print("=" * 70)
    print(f"最佳参数:")
    print(f"  n_estimators: {best_params['n_estimators']}")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  learning_rate: {best_params['learning_rate']}")
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