<div align="center">

# <img src="assets/logo.svg" alt="优图 Logo" height="26px"> Youtu-Embedding: <br> 基于协同-差异化学习的先进统一文本表示模型

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Latest-blue.svg)](https://arxiv.org/abs/2508.11442)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-embedding)
[![Huggingface](https://img.shields.io/badge/Huggingface-YoutuRAG-blue)](https://huggingface.co/tencent/Youtu-Embedding)
[![WeChat Community](https://img.shields.io/badge/Community-WeChat-32CD32)](assets/wechat_qr.png)
[![Discord Community](https://img.shields.io/badge/Community-Discord-8A2BE2)](https://discord.gg/QjqhkHQVVM)


[🔖 English Version](README.md) • [🤗 模型下载](#download) • [🚀 快速开始推理](#quickstart) • [🛠️ 如何训练](#train)

</div>

## 🎯 简介

<img src="assets/rag_logo.png" alt="Youtu-RAG Logo" width="140" align="left" style="margin-right:20px;">

**Youtu-Embedding** 是一款由腾讯优图实验室研发的业界领先的通用文本表示模型。它在信息检索（IR）、语义相似度（STS）、聚类、重排序和分类等一系列广泛的自然语言处理任务上，均展现出卓越的性能。

Youtu-Embedding核心优势包括：

  - **🏆 顶尖性能**: 在权威的中文文本嵌入评测基准 CMTEB 上，以 **77.46**  的高分荣登榜首（截至2025年09月），证明了其强大的表征能力。

  - **🧠 精密的三阶段训练**: 通过“LLM基础预训练 → 弱监督对齐 → 协同-判别式微调”的训练流程，系统性地将大模型的广博知识转化为专用于嵌入任务的判别能力。

  - **⭐ 创新的微调框架**: 设计了统一数据格式、任务差异化损失函数和动态单任务采样机制，解决了多任务学习中的“负迁移”难题，实现了多任务的稳定协同训练。（该框架在多种基础编码器上进行了验证，保障其通用性和有效性）

  - **🛠️ 精细化的数据工程**: 结合了基于LLM的高质量数据合成技术与高效的难负例挖掘策略，为模型训练提供了最坚实的数据基础。

我们在此开源模型权重、推理代码及完整的训练框架，希望能助力社区开发者创造更大的价值！

<a id="download"></a>

 ## 🤗 模型下载

我们已在 Hugging Face 上发布了首个模型版本。这是一个拥有20亿（2B）参数的通用语义表示模型。

| 模型                | 参数量 | 维度   | 序列长度 | Hugging Face  |
| :------------------- | :--------: | :--------: | :-----------------: | :------------------------------------------------------------------------------------------ |
| Youtu-Embedding-V1   | 2B         | 2048       | 8K               | [Model](https://huggingface.co/tencent/Youtu-Embedding) |

<a id="quickstart"></a>

## 🚀 快速开始推理

您可以通过两种方式来生成文本嵌入（Embeddings）：便捷的官方 API 调用，或在本地环境中完全控制地运行模型。

### 选项 1：☁️ 使用官方 API

**📦 安装 SDK**

```bash
pip install --upgrade tencentcloud-sdk-python
```

  * **API 指南**：有关身份验证和终端节点的详细信息，请参阅 [腾讯云 API 文档](https://cloud.tencent.com/document/product/1772/115343)。
  * **SDK 参考**：有关 SDK 的更多信息，请参阅 [SDK 安装指南](https://cloud.tencent.com/document/sdk)。

**⚙️ 使用方法**

  * 请参见 [`usage/tencent_cloud_api.py`](usage/tencent_cloud_api.py) 中的脚本。

### 选项 2：💻 本地化自托管推理

在您自己的机器上运行模型可以赋予您完全的控制权，非常适合离线使用、自定义或数据隐私优先的场景。以下是几种主流的入门方法。

#### 1. 使用自定义 `LLMEmbeddingModel` 类

如果需要更专业的实现或查看我们的直接封装，您可以使用 `LLMEmbeddingModel` 类。

  * 请在此处查看完整的示例脚本：[`usage/infer_llm_embedding.py`](usage/infer_llm_embedding.py)。

#### 2. 使用 `sentence-transformers`

**📦 安装**

```bash
pip install sentence-transformers==5.1.0
```

**⚙️ 使用方法**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("model_id")
queries = ["天气怎么样？"]
passages = [
    '今天天气很好。',
    '外面阳光明媚！',
    '他开车去了体育场。'
]
queries_embeddings = model.encode_query(queries)
passages_embeddings = model.encode_document(passages)

similarities = model.similarity(queries_embeddings, passages_embeddings)
print(similarities)
```

#### 3. 使用 `LangChain` 🦜

轻松将模型集成到您的 **LangChain** 应用中，例如 RAG（检索增强生成）管道。

**📦 安装**

```bash
pip install langchain==0.3.27 langchain-community==0.3.29 langchain-huggingface==0.3.1 sentence-transformers==5.1.0 faiss-cpu==1.11.0
```

**⚙️ 使用方法**

  * 请参阅此示例：[`usage/langchain_embedding.py`](usage/langchain_embedding.py)

#### 4. 使用 `LlamaIndex` 🦙

这非常适合将模型集成到您的 **LlamaIndex** 搜索和检索系统中。

**📦 安装**

```bash
pip install llama-index==0.14.2 llama-index-embeddings-huggingface==0.6.1 sentence-transformers==5.1.0 llama-index-vector-stores-faiss==0.5.1
```

**⚙️ 使用方法**

  * 请参阅此示例：[`usage/llamaindex_embedding.py`](usage/llamaindex_embedding.py)


## 💡 微调训练框架

我们提供的**协同-判别式微调训练框架**，旨在克服不同文本嵌入任务联合优化的挑战。通过系统地解耦任务实现了统一表示学习。

<strong/>🌐 1. 统一且可扩展的数据格式</strong>

我们设计的统一数据结构能够无缝处理来自 IR、STS、分类、重排序等任务的异构数据，为未来接入新任务提供了极高的可扩展性。

<strong/>🎯 2. 任务差异化的损失函数</strong>

我们摒弃了“一刀切”的损失函数，为不同任务设计了专属的优化目标。

  - **对于 IR (信息检索) 类任务**，它采用了一个强大的 InfoNCE 对比损失，支持多正例和困难负例，并结合跨设备负采样以实现更清晰的区分度。
  - **对于 STS (语义相似度) 类任务**，它超越了简单的对比学习方法，转而使用一套排序感知目标来直接优化排序一致性，包括皮尔逊损失、归一化排序KL散度损失。

<strong/>🔄 3. 动态单任务采样</strong>

为避免混合任务批次带来的梯度干扰，我们实现了定制化的动态采样器。它确保在单次训练迭代中，所有 GPU 处理的都是同一个数据集的不重叠分片，从而为模型提供纯粹、稳定的梯度信号。


<a id="train"></a>

### 🛠️ 如何训练

我们的训练框架的代码位于 [`training/`](training/) 目录下。

#### 1\. 安装

下载项目并安装依赖：

```bash
git clone https://github.com/TencentCloudADP/youtu-embedding.git
cd training/CoDiEmb
pip install -r requirements.txt
```

#### 2\. 训练

```bash
cd scripts 
bash train_youtuemb.sh
```

#### 3\. 评估
重现结果的代买可以在[`evaluation/`](evaluation/)部分找到。


#### 📊 CMTEB
| 模型                   |  参数量 | 平均分(任务)         | 平均分(类型)         | 分类   | 聚类   | 句子对分类 | 重排序  | 检索   | 语义相似度 |
| :------------------------ | :--------------| :----------------- | :----------------- | :----: | :----: | :---------: | :-----: | :----: | :---: |
| bge-multilingual-gemma2  | 9B | 67.64              | 68.52              | 75.31  | 59.30  | 79.30      | 68.28   | 73.73  | 55.19 |
| ritrieve\_zh\_v1       | 326M   | 72.71              | 73.85              | 76.88  | 66.50  | 85.98       | 72.86   | 76.97  | 63.92 |
| Qwen3-Embedding-4B      | 4B | 72.27              | 73.51              | 75.46  | 77.89  | 83.34       | 66.05   | 77.03  | 61.26 |
| Qwen3-Embedding-8B     | 8B | 73.84              | 75.00              | 76.97  | 80.08  | 84.23       | 66.99   | 78.21  | 63.53 |
| Conan-embedding-v2      | 1.4B | 74.24              | 75.99              | 76.47  | 68.84  | 92.44       | 74.41   | 78.31  | 65.48 |
| Seed1.6-embedding       | - | 75.63              | 76.68              | 77.98  | 73.11  | 88.71       | 71.65   | 79.69  | 68.94 |
| QZhou-Embedding         | 7B | 76.99              | 78.58              | 79.99  | 70.91  | 95.07       | 74.85   | 78.80  | 71.89 |
| **Youtu-Embedding-V1** | 2B | **77.60** | **78.85** | 78.04 | 79.67 | 89.69 | 73.85 | 80.95 | 70.91 |

> **注意**: 各模型分数来自2025年9月28日记录的MTEB[榜单](https://huggingface.co/spaces/mteb/leaderboard)


## 🎉 引用

如果您在您的研究中使用了我们的工作，请考虑引用我们的论文：

```bibtex
@misc{zhang2025codiemb,
  title={CoDiEmb: A Collaborative yet Distinct Framework for Unified Representation Learning in Information Retrieval and Semantic Textual Similarity},
  author={Zhang, Bowen and Song, Zixin and Chen, Chunquan and Zhang, Qian-Wen and Yin, Di and Sun, Xing},
  year={2025},
  eprint={2508.11442},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2508.11442},
}
```
