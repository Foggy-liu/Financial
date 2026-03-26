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
    """XGBoost分阶段超参数调优（核心参数+正则化参数）"""

    # ============ 第一阶段：调优核心参数 ============
    print("\n" + "=" * 60)
    print("2. XGBoost超参数调优 - 第一阶段：核心参数")
    print("=" * 60)

    n_estimators_list = [100, 200, 300]  # 树的数量
    max_depth_list = [3, 6, 9]  # 树的最大深度
    learning_rate_list = [0.1, 0.3]  # 学习率

    stage1_total = len(n_estimators_list) * len(max_depth_list) * len(learning_rate_list)
    print(f"测试核心参数组合... 共 {stage1_total} 种组合")

    stage1_results = []
    combination_count = 0

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for learning_rate in learning_rate_list:
                combination_count += 1
                if combination_count % 5 == 0:
                    print(f"  已测试 {combination_count}/{stage1_total} 个组合")

                model = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    objective='multi:softprob',
                    num_class=3,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='mlogloss'
                )

                # 交叉验证
                f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
                auc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc_ovr')
                accuracy_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

                stage1_results.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'F1': np.mean(f1_scores),
                    'AUC': np.mean(auc_scores),
                    'Accuracy': np.mean(accuracy_scores)
                })

    # 转换为DataFrame并找出最佳
    stage1_df = pd.DataFrame(stage1_results)
    best_stage1 = stage1_df.loc[stage1_df['F1'].idxmax()]

    print("\n" + "-" * 40)
    print("第一阶段最佳核心参数:")
    print(f"  n_estimators: {int(best_stage1['n_estimators'])}")
    print(f"  max_depth: {int(best_stage1['max_depth'])}")
    print(f"  learning_rate: {best_stage1['learning_rate']}")
    print(f"  F1: {best_stage1['F1']:.4f}")
    print("-" * 40)

    # ============ 第二阶段：固定核心参数，调优正则化 ============
    print("\n" + "=" * 60)
    print("2. XGBoost超参数调优 - 第二阶段：正则化参数")
    print("=" * 60)

    reg_lambda_list = [0.1, 1, 10]  # L2正则化
    reg_alpha_list = [0, 0.1, 1]  # L1正则化

    stage2_total = len(reg_lambda_list) * len(reg_alpha_list)
    print(f"测试正则化参数组合... 共 {stage2_total} 种组合")
    print(f"固定核心参数: n_estimators={int(best_stage1['n_estimators'])}, "
          f"max_depth={int(best_stage1['max_depth'])}, "
          f"learning_rate={best_stage1['learning_rate']}")

    stage2_results = []
    combination_count = 0

    for reg_lambda in reg_lambda_list:
        for reg_alpha in reg_alpha_list:
            combination_count += 1
            print(f"  已测试 {combination_count}/{stage2_total} 个组合: "
                  f"reg_lambda={reg_lambda}, reg_alpha={reg_alpha}")

            model = XGBClassifier(
                n_estimators=int(best_stage1['n_estimators']),
                max_depth=int(best_stage1['max_depth']),
                learning_rate=best_stage1['learning_rate'],
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )

            # 交叉验证
            f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
            auc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc_ovr')
            accuracy_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

            # 训练模型计算测试集交叉熵
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)
            ce_score = log_loss(y_test, y_pred_proba)

            stage2_results.append({
                'reg_lambda': reg_lambda,
                'reg_alpha': reg_alpha,
                'F1': np.mean(f1_scores),
                'AUC': np.mean(auc_scores),
                'Accuracy': np.mean(accuracy_scores),
                'CrossEntropy': ce_score
            })

    # 转换为DataFrame并找出最佳
    stage2_df = pd.DataFrame(stage2_results)
    best_stage2 = stage2_df.loc[stage2_df['F1'].idxmax()]

    print("\n" + "-" * 40)
    print("第二阶段最佳正则化参数:")
    print(f"  reg_lambda: {best_stage2['reg_lambda']}")
    print(f"  reg_alpha: {best_stage2['reg_alpha']}")
    print(f"  F1: {best_stage2['F1']:.4f}")
    print("-" * 40)

    # ============ 合并结果，绘制调优过程图 ============
    print("\n" + "=" * 60)
    print("3. 绘制调优过程可视化")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1：核心参数 - n_estimators的影响
    ax1 = axes[0, 0]
    for max_depth in max_depth_list:
        subset = stage1_df[stage1_df['max_depth'] == max_depth]
        if not subset.empty:
            for lr in learning_rate_list:
                sub_subset = subset[subset['learning_rate'] == lr]
                if not sub_subset.empty:
                    avg_f1 = sub_subset.groupby('n_estimators')['F1'].mean()
                    ax1.plot(avg_f1.index, avg_f1.values, 'o-',
                             label=f'max_depth={max_depth}, lr={lr}')
    ax1.set_xlabel('n_estimators (树的数量)')
    ax1.set_ylabel('F1分数')
    ax1.set_title('核心参数：树数量对F1分数的影响')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 图2：核心参数 - max_depth的影响
    ax2 = axes[0, 1]
    for n_estimators in n_estimators_list:
        subset = stage1_df[stage1_df['n_estimators'] == n_estimators]
        if not subset.empty:
            for lr in learning_rate_list:
                sub_subset = subset[subset['learning_rate'] == lr]
                if not sub_subset.empty:
                    avg_f1 = sub_subset.groupby('max_depth')['F1'].mean()
                    ax2.plot(avg_f1.index, avg_f1.values, 'o-',
                             label=f'n_est={n_estimators}, lr={lr}')
    ax2.set_xlabel('max_depth (树的最大深度)')
    ax2.set_ylabel('F1分数')
    ax2.set_title('核心参数：最大深度对F1分数的影响')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 图3：正则化参数 - reg_lambda的影响
    ax3 = axes[1, 0]
    for reg_alpha in reg_alpha_list:
        subset = stage2_df[stage2_df['reg_alpha'] == reg_alpha]
        if not subset.empty:
            ax3.plot(subset['reg_lambda'], subset['F1'], 'o-',
                     label=f'reg_alpha={reg_alpha}')
    ax3.set_xlabel('reg_lambda (L2正则化强度)')
    ax3.set_ylabel('F1分数')
    ax3.set_title('正则化参数：L2正则化对F1分数的影响')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4：正则化参数 - reg_alpha的影响
    ax4 = axes[1, 1]
    for reg_lambda in reg_lambda_list:
        subset = stage2_df[stage2_df['reg_lambda'] == reg_lambda]
        if not subset.empty:
            ax4.plot(subset['reg_alpha'], subset['F1'], 'o-',
                     label=f'reg_lambda={reg_lambda}')
    ax4.set_xlabel('reg_alpha (L1正则化强度)')
    ax4.set_ylabel('F1分数')
    ax4.set_title('正则化参数：L1正则化对F1分数的影响')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('XGBoost分阶段超参数调优过程', fontsize=16)
    plt.tight_layout()
    plt.savefig('xgboost_tuning.png', dpi=300, bbox_inches='tight')
    print("调优过程图已保存为 'xgboost_tuning.png'")

    try:
        plt.show()
    except:
        print("图形显示失败，但已保存图片")
        plt.close()

    # ============ 输出最终结果 ============
    print("\n" + "=" * 60)
    print("4. 最终最佳参数组合")
    print("=" * 60)

    best_params = {
        'n_estimators': int(best_stage1['n_estimators']),
        'max_depth': int(best_stage1['max_depth']),
        'learning_rate': best_stage1['learning_rate'],
        'reg_lambda': best_stage2['reg_lambda'],
        'reg_alpha': best_stage2['reg_alpha']
    }

    print(f"  n_estimators: {best_params['n_estimators']}")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  reg_lambda: {best_params['reg_lambda']}")
    print(f"  reg_alpha: {best_params['reg_alpha']}")
    print(f"\n  交叉验证F1: {best_stage2['F1']:.4f}")
    print(f"  交叉验证AUC: {best_stage2['AUC']:.4f}")
    print(f"  交叉验证准确率: {best_stage2['Accuracy']:.4f}")

    # 显示Top 5正则化组合
    print("\nTop 5 正则化参数组合 (按F1排序):")
    top5 = stage2_df.nlargest(5, 'F1')[['reg_lambda', 'reg_alpha', 'F1', 'AUC', 'Accuracy']]
    print(top5.to_string(index=False))

    return best_params, stage2_df


def train_final_model(X_train, y_train, X_test, y_test, best_params):
    """使用最佳参数训练最终模型（对比有无类别平衡）"""
    print("\n" + "=" * 60)
    print("5. 最终模型训练与评估")
    print("=" * 60)

    target_names = ['负面(0)', '中性(1)', '正面(2)']

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

        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_per_class = f1_score(y_test, y_pred, average=None)

        print(f"准确率: {accuracy:.4f}")
        print(f"AUC分数: {auc:.4f}")
        print(f"交叉熵损失: {ce:.4f}")
        print(f"F1宏观平均: {f1_macro:.4f}")
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

    if results['平衡权重']['f1_macro'] > results['无权重']['f1_macro']:
        print("\n平衡权重模型表现更好（F1宏观平均更高），选择此模型")
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
        print("\n无权重模型表现更好（F1宏观平均更高），选择此模型")
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
    print("XGBoost模型训练与超参数调优（分阶段调优：核心参数+正则化）")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_data()

    best_params, tuning_results = tune_xgboost_hyperparameters(X_train, y_train, X_test, y_test)

    final_model, accuracy, auc, ce, f1_macro, f1_weighted, f1_per_class, weight_used = train_final_model(
        X_train, y_train, X_test, y_test, best_params
    )

    print("\n" + "=" * 60)
    print("6. 保存结果")
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
    print(f"  reg_lambda: {best_params['reg_lambda']}")
    print(f"  reg_alpha: {best_params['reg_alpha']}")
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