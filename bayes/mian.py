import matplotlib

matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
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


def preprocess_for_nb(X_train, X_test, nb_type):
    """
    根据朴素贝叶斯类型对数据进行预处理

    Parameters:
    -----------
    nb_type: 'gaussian', 'multinomial', 'bernoulli'
    """
    if nb_type == 'gaussian':
        # 高斯朴素贝叶斯：不需要特殊处理
        return X_train.copy(), X_test.copy()

    elif nb_type == 'multinomial':
        # 多项式朴素贝叶斯：需要非负特征
        # 将数据平移到非负区间
        min_val = X_train.min()
        if min_val < 0:
            X_train_pos = X_train - min_val
            X_test_pos = X_test - min_val
        else:
            X_train_pos = X_train.copy()
            X_test_pos = X_test.copy()

        # 确保是整数（多项式朴素贝叶斯通常用于计数）
        X_train_pos = np.round(X_train_pos).astype(int)
        X_test_pos = np.round(X_test_pos).astype(int)

        # 确保没有负数
        X_train_pos = np.maximum(X_train_pos, 0)
        X_test_pos = np.maximum(X_test_pos, 0)

        return X_train_pos, X_test_pos

    elif nb_type == 'bernoulli':
        # 伯努利朴素贝叶斯：需要二值化
        # 以0为阈值进行二值化
        X_train_bin = (X_train > 0).astype(int)
        X_test_bin = (X_test > 0).astype(int)
        return X_train_bin, X_test_bin

    else:
        raise ValueError(f"未知的朴素贝叶斯类型: {nb_type}")


def tune_naivebayes_hyperparameters(X_train, y_train, X_test, y_test):
    """朴素贝叶斯超参数调优"""
    print("\n" + "=" * 60)
    print("2. 朴素贝叶斯超参数调优")
    print("=" * 60)

    # 三种朴素贝叶斯类型
    nb_types = ['gaussian', 'multinomial', 'bernoulli']

    # 参数范围
    var_smoothing_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # 高斯朴素贝叶斯的平滑参数
    alpha_list = [0.1, 0.5, 1.0, 2.0, 5.0]  # 多项式/伯努利朴素贝叶斯的平滑参数

    results_df = pd.DataFrame()

    print("测试不同朴素贝叶斯类型和参数组合...")

    for nb_type in nb_types:
        print(f"\n  测试 {nb_type} 类型...")

        if nb_type == 'gaussian':
            param_list = var_smoothing_list
            param_name = 'var_smoothing'
        else:
            param_list = alpha_list
            param_name = 'alpha'

        for param_value in param_list:
            # 根据类型创建模型
            if nb_type == 'gaussian':
                model = GaussianNB(var_smoothing=param_value)
            elif nb_type == 'multinomial':
                model = MultinomialNB(alpha=param_value)
            else:  # bernoulli
                model = BernoulliNB(alpha=param_value)

            # 数据预处理
            X_train_proc, X_test_proc = preprocess_for_nb(X_train, X_test, nb_type)

            # 交叉验证
            accuracy_scores = cross_val_score(model, X_train_proc, y_train, cv=5,
                                              scoring='accuracy')

            # 训练模型
            model.fit(X_train_proc, y_train)

            # 预测概率
            y_pred_proba = model.predict_proba(X_test_proc)

            # 计算AUC
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                auc = np.nan

            # 计算交叉熵
            ce = log_loss(y_test, y_pred_proba)

            # 预测标签用于F1计算
            y_pred = model.predict(X_test_proc)
            f1_macro = f1_score(y_test, y_pred, average='macro')

            temp_df = pd.DataFrame({
                'nb_type': [nb_type],
                'param_name': [param_name],
                'param_value': [param_value],
                'Accuracy': [np.mean(accuracy_scores)],
                'AUC': [auc],
                'F1': [f1_macro],
                'CrossEntropy': [ce]
            })
            results_df = pd.concat([results_df, temp_df], ignore_index=True)

    print("\n参数测试完成！")

    # 绘制调优过程
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 图1：不同NB类型的F1对比
    ax1 = axes[0, 0]
    for idx, nb_type in enumerate(nb_types):
        subset = results_df[results_df['nb_type'] == nb_type]
        best_f1 = subset['F1'].max()
        ax1.bar(nb_type, best_f1, color=colors[idx], alpha=0.7)
        ax1.text(idx, best_f1 + 0.005, f'{best_f1:.4f}', ha='center')
    ax1.set_ylabel('F1分数')
    ax1.set_title('不同类型朴素贝叶斯的最佳F1对比')
    ax1.grid(True, alpha=0.3)

    # 图2：高斯NB的平滑参数影响
    ax2 = axes[0, 1]
    gaussian_subset = results_df[results_df['nb_type'] == 'gaussian'].sort_values('param_value')
    ax2.plot(gaussian_subset['param_value'], gaussian_subset['F1'],
             'o-', color=colors[0], linewidth=2)
    ax2.set_xscale('log')
    ax2.set_xlabel('var_smoothing')
    ax2.set_ylabel('F1分数')
    ax2.set_title('高斯朴素贝叶斯 - 平滑参数影响')
    ax2.grid(True, alpha=0.3)

    # 图3：多项式NB的alpha参数影响
    ax3 = axes[1, 0]
    multinomial_subset = results_df[results_df['nb_type'] == 'multinomial'].sort_values('param_value')
    ax3.plot(multinomial_subset['param_value'], multinomial_subset['F1'],
             's-', color=colors[1], linewidth=2)
    ax3.set_xlabel('alpha')
    ax3.set_ylabel('F1分数')
    ax3.set_title('多项式朴素贝叶斯 - alpha参数影响')
    ax3.grid(True, alpha=0.3)

    # 图4：伯努利NB的alpha参数影响
    ax4 = axes[1, 1]
    bernoulli_subset = results_df[results_df['nb_type'] == 'bernoulli'].sort_values('param_value')
    ax4.plot(bernoulli_subset['param_value'], bernoulli_subset['F1'],
             '^-', color=colors[2], linewidth=2)
    ax4.set_xlabel('alpha')
    ax4.set_ylabel('F1分数')
    ax4.set_title('伯努利朴素贝叶斯 - alpha参数影响')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('朴素贝叶斯超参数调优过程', fontsize=16)
    plt.tight_layout()
    plt.savefig('naivebayes_tuning.png', dpi=300, bbox_inches='tight')
    print("调优过程图已保存为 'naivebayes_tuning.png'")

    try:
        plt.show()
    except:
        print("图形显示失败，但已保存图片")
        plt.close()

    # 找出最佳参数
    best_idx = results_df['F1'].idxmax()
    best_params_row = results_df.loc[best_idx]

    print("\n" + "=" * 60)
    print("最佳参数组合 (基于F1分数):")
    print(f"  朴素贝叶斯类型: {best_params_row['nb_type']}")
    print(f"  参数名: {best_params_row['param_name']}")
    print(f"  参数值: {best_params_row['param_value']}")
    print(f"  F1: {best_params_row['F1']:.4f} ⭐")
    print(f"  AUC: {best_params_row['AUC']:.4f}")
    print(f"  Accuracy: {best_params_row['Accuracy']:.4f}")

    # 显示Top 3最佳组合
    print("\nTop 3 最佳参数组合 (按F1排序):")
    top3 = results_df.nlargest(3, 'F1')[['nb_type', 'param_name', 'param_value', 'F1', 'AUC']]
    print(top3.to_string(index=False))

    best_params = {
        'nb_type': best_params_row['nb_type'],
        'param_name': best_params_row['param_name'],
        'param_value': best_params_row['param_value']
    }

    return best_params, results_df


def train_final_model(X_train, y_train, X_test, y_test, best_params):
    """使用最佳参数训练最终模型"""
    print("\n" + "=" * 60)
    print("3. 最终模型训练与评估")
    print("=" * 60)

    target_names = ['负面(0)', '中性(1)', '正面(2)']

    # 根据最佳参数创建模型
    nb_type = best_params['nb_type']
    param_value = best_params['param_value']

    # 数据预处理
    X_train_proc, X_test_proc = preprocess_for_nb(X_train, X_test, nb_type)

    # 计算类别权重（用于样本加权）
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights_balanced = compute_sample_weight('balanced', y_train)

    # 训练两个模型
    models = {}

    # 无权重模型
    if nb_type == 'gaussian':
        models['无权重'] = GaussianNB(var_smoothing=param_value)
    elif nb_type == 'multinomial':
        models['无权重'] = MultinomialNB(alpha=param_value)
    else:  # bernoulli
        models['无权重'] = BernoulliNB(alpha=param_value, binarize=0.0)

    # 平衡权重模型（通过样本加权实现）
    if nb_type == 'gaussian':
        models['平衡权重'] = GaussianNB(var_smoothing=param_value)
    elif nb_type == 'multinomial':
        models['平衡权重'] = MultinomialNB(alpha=param_value)
    else:  # bernoulli
        models['平衡权重'] = BernoulliNB(alpha=param_value, binarize=0.0)

    results = {}

    for name, model in models.items():
        print(f"\n--- {name}模型 (类型: {nb_type}) ---")

        # 训练（平衡权重模型使用样本加权）
        if name == '平衡权重':
            model.fit(X_train_proc, y_train, sample_weight=sample_weights_balanced)
        else:
            model.fit(X_train_proc, y_train)

        # 预测
        y_pred = model.predict(X_test_proc)
        y_pred_proba = model.predict_proba(X_test_proc)

        # 评估指标
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            auc = np.nan
        ce = log_loss(y_test, y_pred_proba)

        # F1分数
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
    title = f'混淆矩阵 - {nb_type}最终模型'
    if weight_used:
        title += ' (带类别平衡)'
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('naivebayes_confusion_matrix.png', dpi=300)

    try:
        plt.show()
    except:
        print("混淆矩阵显示失败，但已保存图片")
        plt.close()

    return final_model, final_accuracy, final_auc, final_ce, final_f1_macro, final_f1_weighted, final_f1_per_class, weight_used, nb_type


def main():
    """主函数"""
    print("=" * 70)
    print("朴素贝叶斯模型训练与超参数调优")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_data()

    best_params, tuning_results = tune_naivebayes_hyperparameters(X_train, y_train, X_test, y_test)

    final_model, accuracy, auc, ce, f1_macro, f1_weighted, f1_per_class, weight_used, nb_type = train_final_model(
        X_train, y_train, X_test, y_test, best_params
    )

    print("\n" + "=" * 60)
    print("4. 保存结果")
    print("=" * 60)

    model_info = {
        'model': final_model,
        'best_params': best_params,
        'class_weight': weight_used,
        'accuracy': accuracy,
        'auc': auc,
        'cross_entropy': ce,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class.tolist() if f1_per_class is not None else None
    }
    joblib.dump(model_info, 'best_naivebayes_model.pkl')
    print("模型已保存为 'best_naivebayes_model.pkl'")

    tuning_results.to_csv('naivebayes_tuning_results.csv', index=False)
    print("调优结果已保存为 'naivebayes_tuning_results.csv'")

    print("\n" + "=" * 70)
    print("训练完成！最终结果总结")
    print("=" * 70)
    print(f"最佳参数:")
    print(f"  朴素贝叶斯类型: {best_params['nb_type']}")
    print(f"  参数名: {best_params['param_name']}")
    print(f"  参数值: {best_params['param_value']}")
    print(f"类别平衡: {'使用样本加权' if weight_used else '未使用'}")
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