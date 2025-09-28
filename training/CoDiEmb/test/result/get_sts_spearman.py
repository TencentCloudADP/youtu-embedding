import os
import json

def analyze_sts_results(folder_path):
    """
    读取指定文件夹中的所有 STS 任务 .json 文件，提取 Spearman 相关系数，并计算总分和平均分。

    此版本会优先查找 "validation" 字段，如果找不到，则会查找 "test" 字段。

    :param folder_path: 存放 .json 文件的文件夹路径
    """
    scores = []
    results = []

    assert os.path.isdir(folder_path), f"错误：文件夹 '{folder_path}' 不存在。"

    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith('.json'):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset_name = data.get("mteb_dataset_name")

                # 首先尝试从 "validation" 中提取分数
                spearman_score = data.get("validation", {}).get("cos_sim", {}).get("spearman")
                # 如果从 "validation" 中没有取到值 (即 spearman_score 为 None), 则尝试从 "test" 中提取
                assert spearman_score is not None
                spearman_score = data.get("test", {}).get("cos_sim", {}).get("spearman")

                assert (dataset_name and spearman_score is not None), \
                    f"警告：文件 '{filename}' 中未找到 'validation/cos_sim/spearman' 或 'test/cos_sim/spearman' 字段。"

                scores.append(spearman_score * 100)
                percentage_score = round(spearman_score * 100, 2)
                results.append(f"{dataset_name} : {spearman_score} => {percentage_score:.2f}")

        except json.JSONDecodeError:
            print(f"警告：无法解析文件 '{filename}'。")

    # 注意：这里的 total 和 avg 是基于百分制分数计算的
    total_score_percentage = sum(scores)
    avg_score_percentage = total_score_percentage / len(scores)

    print(f"\ntotal: {total_score_percentage:.2f}")
    print(f"avg: {avg_score_percentage:.2f}")


def main():
    # --- 执行分析 ---
    folder_path = './STS/model_MiniCPM-Embedding'
    print(f"--- 分析文件夹: {folder_path} ---")
    analyze_sts_results(folder_path)


if __name__ == '__main__':
    main()
