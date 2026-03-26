import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train_xgboost(X_train, y_train, X_test, y_test, cv_folds=5):
    """
    使用XGBoost进行训练
    """
    print("\n" + "=" * 70)
    print(f"XGBoost训练 - {cv_folds}折交叉验证")
    print("=" * 70)

    # 参数设置
    n_estimators_list = [50, 100, 200]
    max_depth_list = [3, 6, 9]
    learning_rate_list = [0.01, 0.1, 0.3]

    results = []

    print("\n开始网格搜索...")
    total_combos = len(n_estimators_list) * len(max_depth_list) * len(learning_rate_list)
    print(f"参数组合: {len(n_estimators_list)}个树数量 × {len(max_depth_list)}个深度 × {len(learning_rate_list)}个学习率 = {total_combos}种")
    print("-" * 50)

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for learning_rate in learning_rate_list:
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
                auc_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc_ovr')
                f1_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_macro')
                accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

                mean_auc = np.mean(auc_scores)
                mean_f1 = np.mean(f1_scores)
                mean_accuracy = np.mean(accuracy_scores)

                print(f"  n={n_estimators}, depth={max_depth}, lr={learning_rate}: AUC={mean_auc:.4f}, F1={mean_f1:.4f}")

                results.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
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
    print(f"  n_estimators = {int(best['n_estimators'])}")
    print(f"  max_depth = {int(best['max_depth'])}")
    print(f"  learning_rate = {best['learning_rate']}")
    print(f"  交叉验证AUC = {best['auc']:.4f}")
    print(f"  交叉验证F1 = {best['f1']:.4f}")
    print(f"  交叉验证准确率 = {best['accuracy']:.4f}")
    print("=" * 50)

    # 用最佳参数训练最终模型
    best_params = {
        'n_estimators': int(best['n_estimators']),
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }

    final_model = XGBClassifier(**best_params)
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

    # 特征重要性分析
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 6))
    plt.bar(range(20), importances[indices])
    plt.xticks(range(20), [f'PC{i + 1}' for i in indices], rotation=45)
    plt.xlabel('PCA特征')
    plt.ylabel('特征重要性')
    plt.title('XGBoost特征重要性排名 (Top 20)')
    plt.tight_layout()
    plt.savefig('xgb_feature_importance.png', dpi=300)
    plt.show()

    return final_model, results_df, test_auc, test_f1_macro, test_accuracy


def main():
    """主函数"""
    print("=" * 70)
    print("XGBoost模型训练")
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

    # 2. 训练XGBoost
    model, results, test_auc, test_f1, test_acc = train_xgboost(
        X_train, y_train, X_test, y_test, cv_folds=5
    )

    # 3. 绘制混淆矩阵
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面(0)', '中性(1)', '正面(2)'],
                yticklabels=['负面(0)', '中性(1)', '正面(2)'],
                annot_kws={'size': 14})

    plt.title('混淆矩阵 - XGBoost (SMOTE后)', fontsize=14)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig('xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n混淆矩阵已保存为 'xgb_confusion_matrix.png'")

    # 4. 保存模型
    joblib.dump(model, 'best_xgboost_model_smote.pkl')
    print(f"模型已保存为 'best_xgboost_model_smote.pkl'")

    # 5. 保存结果
    results.to_csv('xgb_tuning_results.csv', index=False)
    print("调优结果已保存为 'xgb_tuning_results.csv'")

    print("\n" + "=" * 70)
    print("XGBoost训练完成！")
    print("=" * 70)
    print(f"最佳参数: n_estimators={model.n_estimators}, max_depth={model.max_depth}, learning_rate={model.learning_rate}")
    print(f"测试集AUC: {test_auc:.4f}")
    print(f"测试集F1宏观平均: {test_f1:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()