# 快速开始

本节提供最小示例，帮助你快速体验检索功能。
输入 Query 后，系统会基于向量相似度返回 Top-K 结果。

## 高级用法
可通过 --top_k 指定返回的结果数；通过 --strip_frontmatter 去除 YAML 头部信息。

### 故障排查
如果遇到错误，请检查模型路径 ./Youtu-Embedding 是否可用，以及 test-docs 下是否存在 .md 文件。