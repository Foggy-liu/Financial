import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def compare_all_models(all_results):
    """
    对比所有模型的最终表现（每个模型取最佳权重）
    """
    print("\n" + "=" * 70)
    print("模型最终对比")
    print("=" * 70)

    # 准备对比数据
    comparison = []
    for model_name, model_results in all_results.items():
        # 取最佳权重策略（以F1宏观平均为准）
        if model_results['平衡权重']['f1_macro'] > model_results['无权重']['f1_macro']:
            best_f1 = model_results['平衡权重']['f1_macro']
            best_auc = model_results['平衡权重']['auc']
            best_accuracy = model_results['平衡权重']['accuracy']
            weight_used = '平衡权重'
        else:
            best_f1 = model_results['无权重']['f1_macro']
            best_auc = model_results['无权重']['auc']
            best_accuracy = model_results['无权重']['accuracy']
            weight_used = '无权重'

        comparison.append({
            '模型': model_name,
            'F1宏观平均': best_f1,
            'AUC': best_auc,
            '准确率': best_accuracy,
            '选用权重': weight_used
        })

    # 转成DataFrame并按F1宏观平均排序（从高到低）
    import pandas as pd
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('F1宏观平均', ascending=False).reset_index(drop=True)

    # 打印表格
    print("\n模型性能对比表（每个模型取最佳权重）:")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    print("-" * 80)

    # 找出最佳模型
    best_model = comparison_df.iloc[0]
    print(f"\n最佳模型: {best_model['模型']}")
    print(f"  F1宏观平均: {best_model['F1宏观平均']:.4f}")
    print(f"  AUC: {best_model['AUC']:.4f}")
    print(f"  准确率: {best_model['准确率']:.4f}")
    print(f"  选用权重: {best_model['选用权重']}")
    print("=" * 70)

    # 绘制对比柱状图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    models = comparison_df['模型'].tolist()
    f1_scores = comparison_df['F1宏观平均'].tolist()

    # 最佳模型用深色，其他用浅色
    colors = ['#2c3e50' if i == 0 else '#95a5a6' for i in range(len(models))]

    bars = plt.bar(models, f1_scores, color=colors, alpha=0.8)
    plt.ylabel('F1宏观平均')
    plt.title('各模型最佳表现对比')
    plt.ylim(0, 1)

    # 添加数值标签
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存为 'model_comparison.png'")

    try:
        plt.show()
    except:
        plt.close()

    return best_model['模型'], comparison_df


def collect_all_results():
    """
    收集所有模型的运行结果（示例函数）
    实际使用时，需要从各个模型的训练函数中获取结果
    """
    all_results = {}

    # 示例数据（请替换为实际结果）
    all_results['逻辑回归'] = {
        '无权重': {'f1_macro': 0.3894, 'auc': 0.6268, 'accuracy': 0.5961},
        '平衡权重': {'f1_macro': 0.4362, 'auc': 0.6276, 'accuracy': 0.5754}
    }
    all_results['随机森林'] = {
        '无权重': {'f1_macro': 0.3889, 'auc': 0.6297, 'accuracy': 0.6064},
        '平衡权重': {'f1_macro': 0.3859, 'auc': 0.63511, 'accuracy': 0.6198}
    }
    all_results['XGBoost'] = {
        '无权重': {'f1_macro': 0.4034, 'auc': 0.6148, 'accuracy': 0.5733},
        '平衡权重': {'f1_macro': 0.4159, 'auc': 0.5985, 'accuracy': 0.5351}
    }
    all_results['朴素贝叶斯'] = {
        '无权重': {'f1_macro': 0.4148, 'auc': 0.6080, 'accuracy': 0.6012},
        '平衡权重': {'f1_macro': 0.3573, 'auc': 0.5989, 'accuracy': 0.3678}
    }

    return all_results


# 使用示例
if __name__ == "__main__":
    # 收集所有模型结果
    all_results = collect_all_results()

    # 对比选出最佳模型
    best_model, comparison_df = compare_all_models(all_results)

    print(f"\n最终选择: {best_model}")

    # 可选：保存对比结果到CSV
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print("对比结果已保存为 'model_comparison_results.csv'")