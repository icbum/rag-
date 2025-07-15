# 📚 RAG 知识库问答系统（支持 TXT / PDF / Word）

本项目是一个基于向量检索（RAG）机制的本地知识库问答系统，支持从 TXT、PDF、Word 文件中提取知识，并结合大语言模型（如 ChatGPT）进行智能问答。

---

## ✅ 功能简介

- 支持从 `.txt`、`.pdf`、`.docx` 文档中加载知识内容
- 使用 `SentenceTransformer` 生成句向量
- 使用 `FAISS` 进行相似文本检索
- 支持与多种大模型接口连接（OpenAI、DeepSeek、Qwen 等）
- 可运行在本地环境中，结合自己的私有文档提问
