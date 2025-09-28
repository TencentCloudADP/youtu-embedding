import os
import json


def analyze_retrieval_results(folder_path):
    """
    读取指定文件夹中的所有 .json 文件，提取信息并计算总分和平均分。
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
                # 检查 'dev' 和 'ndcg_at_10' 是否存在
                ndcg_score = data.get("dev", {}).get("ndcg_at_10")

                assert dataset_name and ndcg_score is not None, \
                    f"警告：文件 '{filename}' 缺少 'mteb_dataset_name' 或 'ndcg_at_10' 字段。"

                scores.append(ndcg_score)
                percentage_score = round(ndcg_score * 100, 2)
                results.append(f"{dataset_name} : {ndcg_score} => {percentage_score:.2f}")

        except json.JSONDecodeError:
            print(f"警告：无法解析文件 '{filename}'。")

    assert len(scores) > 0, "未找到任何有效的数据进行统计。"
    total_score = sum(scores) * 100
    avg_score = total_score / len(scores)
    print(f"\ntotal: {total_score:.2f}")
    print(f"avg: {avg_score:.2f}")


def main():
    # --- 执行分析 ---
    folder_path = './IR/model_MiniCPM-Embedding'
    print(f"--- 分析文件夹: {folder_path} ---")
    analyze_retrieval_results(folder_path=folder_path)


if __name__ == '__main__':
    main()
