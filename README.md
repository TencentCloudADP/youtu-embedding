<div align="center">

# <img src="assets/logo.svg" alt="Youtu Logo" height="26px"> Youtu-Embedding: <br> Advancing Unified Text Representation with Collaborative-Distinct Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Latest-blue.svg)](https://arxiv.org/abs/2508.11442)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-embedding)
[![Huggingface](https://img.shields.io/badge/Huggingface-Youtu-blue)](https://huggingface.co/tencent/Youtu-Embedding)
[![WeChat Community](https://img.shields.io/badge/Community-WeChat-32CD32)](assets/wechat_qr.png)
[![Discord Community](https://img.shields.io/badge/Community-Discord-8A2BE2)](https://discord.gg/dwHuBUKkxw)


[🔖 中文版](README-CN.md) • [🤗 Model Download](#download) • [🚀 Quickly Start Inference](#quickstart) • [🛠️ How to Train](#train)

</div>

## 🎯 Brief Introduction

<img src="assets/rag_logo.png" alt="Youtu-RAG Logo" width="140" align="left" style="margin-right:20px;">

**Youtu-Embedding** is an industry-leading, general-purpose text representation model developed by Tencent Youtu Lab. It demonstrates state-of-the-art performance across a wide range of natural language processing tasks, including Information Retrieval (IR), Semantic Textual Similarity (STS), Clustering, Reranking, and Classification.

The core advantages of Youtu-Embedding can be summarized as follows:

  - **🏆 State-of-the-Art Performance**: Achieved a top score of **77.46**  on the authoritative Chinese text embedding benchmark CMTEB (as of Sep 2025), proving its powerful representation capabilities.

  - **🧠 Sophisticated Three-Stage Training:** We pioneered a "LLM-based Pre-training → Weakly-supervised Alignment → Collaborative-Discriminative Fine-tuning" pipeline, which systematically distills the broad knowledge of large language models into the specialized discriminative power required for embedding tasks.

  - **⭐ Innovative Fine-tuning Framework**: We designed a unique Collaborative-Discriminative Fine-tuning Framework that effectively resolves the "negative transfer" problem in multi-task learning through a unified data format, task-differentiated loss functions, and a dynamic single-task sampling mechanism. (This framework has been verified on a variety of basic encoders to ensure its versatility and effectiveness.)

  - **🛠️ Meticulous Data Engineering**: We combined high-quality, LLM-based data synthesis with efficient hard negative mining strategies to provide the most robust data foundation for model training.

We are open-sourcing the model weights, inference code, and the training framework. We hope this will help developers in the community create greater value.

<!-- ## 🗞️ News 🔥
- [2025-09-xx] We release our latest 2B-scale embedding model [Youtu-Embedding-V1](https://huggingface.co/Youtu-RAG)  ! -->

<a id="download"></a>

## 🤗 Model Download

We have released our first model version on Hugging Face. It is a 2 billion (2B) parameter model designed for general-purpose semantic representation.

| Model Name               | Parameters | Dimensions | Sequence Length | Download |
| :------------------- | :--------: | :--------: | :-----------------: | :------------------------------------------------------------------------------------------ |
| Youtu-Embedding-V1   | 2B         | 2048       | 8K               | [Model](https://huggingface.co/tencent/Youtu-Embedding) |



<a id="quickstart"></a>
## 🚀 Quickly Start Inference
You can generate embeddings in two ways: via our official API for ease of use or by running the model locally for full control.

### Option 1: ☁️ Using the Official API
**📦 Install the SDK** 

```bash
pip install --upgrade tencentcloud-sdk-python
```
  * **API Guide**: For details on authentication and endpoints, see the [Tencent Cloud API Documentation](https://cloud.tencent.com/document/product/1772/115343).
  * **SDK Reference**: For more on the SDK, refer to the [SDK Installation Guide](https://cloud.tencent.com/document/sdk).

**⚙️ Usage**

 * Please see the script in [`usage/tencent_cloud_api.py`](usage/tencent_cloud_api.py).

### Option 2: 💻 Locally with Self-Hosted Inference
Running the model on your own machine gives you full control, making it perfect for offline use, customization, or when data privacy is a priority. Here are a few popular ways to get started.

#### 1. Using the Custom `LLMEmbeddingModel` Class
For a more specialized implementation or to see our direct wrapper, you can use the `LLMEmbeddingModel` class.

  * See the complete example script here: [`usage/infer_llm_embedding.py`](usage/infer_llm_embedding.py).

#### 2. Using `sentence-transformers`
**📦 Installation**
```bash
pip install sentence-transformers==5.1.0
```
**⚙️ Usage**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("model_id")
queries = ["What's the weather like?"]
passages = [
    'The weather is lovely today.',
    "It's so sunny outside!",
    'He drove to the stadium.'
]
queries_embeddings = model.encode_query(queries)
passages_embeddings = model.encode_document(passages)

similarities = model.similarity(queries_embeddings, passages_embeddings)
print(similarities)
```

#### 3. Using `LangChain` 🦜
Easily integrate the model into your **LangChain** applications, such as RAG pipelines.

**📦 Installation**

```bash
pip install langchain==0.3.27 langchain-community==0.3.29 langchain-huggingface==0.3.1 sentence-transformers==5.1.0 faiss-cpu==1.11.0
```

**⚙️ Usage**

 * See this example:  [`usage/langchain_embedding.py`](usage/langchain_embedding.py)

#### 4. Using `LlamaIndex` 🦙
This is perfect for integrating the model into your **LlamaIndex** search and retrieval systems.

**📦 Installation**

```bash
pip install llama-index==0.14.2 llama-index-embeddings-huggingface==0.6.1 sentence-transformers==5.1.0 llama-index-vector-stores-faiss==0.5.1
```

**⚙️ Usage**

 * See this example:  [`usage/llamaindex_embedding.py`](usage/llamaindex_embedding.py)



## 💡 Fine-tuning Framework

We provide our novel **Collaborative-Discriminative Fine-tuning Framework**, designed to overcome the challenges of jointly optimizing different text embedding tasks. By systematically decoupling tasks, we introduce several key innovations to achieve highly efficient unified representation learning.

<strong/>🌐 1. Unified & Extensible Data Format</strong>

  Our unified data structure seamlessly handles heterogeneous data from IR, STS, classification, and reranking tasks, offering excellent extensibility for incorporating new tasks in the future.

<strong/>🎯 2. Task-Differentiated Loss Functions</strong>

We moved beyond a "one-size-fits-all" loss function and designed specialized optimization objectives for different tasks.

- **For IR (Information Retrieval) tasks**: We use a powerful InfoNCE contrastive loss that supports multiple positives, hard negatives, and in-batch cross-device negative sampling for superior discriminative ability.

- **For STS (Semantic Textual Similarity) tasks**: We go beyond simple contrastive learning by adopting ranking-aware objectives (e.g., Pearson loss, KL divergence loss L_RankKL) to directly optimize for ranking consistency.

<strong/>🔄 3. Dynamic Single-Task Sampling</strong>

To prevent gradient interference from mixed-task batches, we implemented a custom dynamic sampler. It ensures that within a single training iteration, all GPUs process non-overlapping shards of the same dataset, providing the model with a pure and stable gradient signal.


<a id="train"></a>

### 🛠️ How to Train
The code for our training framework is located in the [`training/`](training/) directory. 

#### 1\. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-embedding.git
cd training/CoDiEmb
pip install -r requirements.txt
```

#### 2\. Training

```bash
cd scripts
bash train_youtuemb.sh
```

#### 3\. Evaluation
The code for reproducing the following results is available in [`evaluation/`](evaluation/).


## 📊 CMTEB

Youtu-Embedding demonstrates superior performance across all seven task categories on the CMTEB benchmark and achieves the highest overall average score. We present the results of the latest version of the model as follows:

| Model                    | Param.  | Mean(Task)         | Mean(Type)         | Class. | Clust. | Pair Class. | Rerank. | Retr.  | STS   |
| :------------------------ | :--------------| :----------------- | :----------------- | :----: | :----: | :---------: | :-----: | :----: | :---: |
| bge-multilingual-gemma2  | 9B | 67.64              | 68.52              | 75.31  | 59.30  | 79.30      | 68.28   | 73.73  | 55.19 |
| ritrieve\_zh\_v1       | 326M   | 72.71              | 73.85              | 76.88  | 66.50  | 85.98       | 72.86   | 76.97  | 63.92 |
| Qwen3-Embedding-4B      | 4B | 72.27              | 73.51              | 75.46  | 77.89  | 83.34       | 66.05   | 77.03  | 61.26 |
| Qwen3-Embedding-8B     | 8B | 73.84              | 75.00              | 76.97  | 80.08  | 84.23       | 66.99   | 78.21  | 63.53 |
| Conan-embedding-v2      | 1.4B | 74.24              | 75.99              | 76.47  | 68.84  | 92.44       | 74.41   | 78.31  | 65.48 |
| Seed1.6-embedding       | - | 75.63              | 76.68              | 77.98  | 73.11  | 88.71       | 71.65   | 79.69  | 68.94 |
| QZhou-Embedding         | 7B | 76.99              | 78.58              | 79.99  | 70.91  | 95.07       | 74.85   | 78.80  | 71.89 |
| **Youtu-Embedding-V1** | 2B | **77.58** | **78.86** | 78.65 | 84.27 | 86.12 | 75.10 | 80.21 | 68.82 |

> **Note**: Comparative scores are based from the MTEB [leaderboard](https://huggingface.co/spaces/mteb/leaderboard), recorded on September 28, 2025.

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 💻 Code Contribution

1. 🍴 Fork the project
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 Create a Pull Request

## 🎉 Citation

If you find our work useful in your research, please consider citing our paper:

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
