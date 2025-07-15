import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF，用于读取PDF
import docx  # python-docx，用于读取Word文档
from api_key import chat_with_model  # 你之前封装的统一调用大模型的接口函数


def load_txt(path):
    """
    读取txt文件，逐行读取非空文本，返回文本列表
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # 去除空行和多余空格
    print(f"✅ 已加载TXT文本，条数：{len(lines)}")
    return lines


def load_pdf(path):
    """
    读取pdf文件，提取所有页的文本内容，按行拆分，返回文本列表
    """
    doc = fitz.open(path)  # 打开PDF文档
    texts = []
    for page in doc:
        text = page.get_text("text")  # 获取当前页纯文本
        # 按行拆分，并去除空行
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        texts.extend(lines)
    print(f"✅ 已加载PDF文本，共提取 {len(texts)} 条内容")
    return texts


def load_word(path):
    """
    读取Word文件，提取所有段落文本，返回文本列表
    """
    doc = docx.Document(path)  # 加载Word文档
    texts = []
    for para in doc.paragraphs:
        line = para.text.strip()
        if line:
            texts.append(line)
    print(f"✅ 已加载Word文本，共提取 {len(texts)} 条内容")
    return texts


def load_documents(path):
    """
    统一读取函数，根据文件后缀判断格式，调用对应函数加载文本
    支持txt、pdf、doc、docx格式
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"❌ 未找到文件：{path}")
    ext = os.path.splitext(path)[1].lower()  # 获取文件后缀
    if ext == ".txt":
        return load_txt(path)
    elif ext == ".pdf":
        return load_pdf(path)
    elif ext in [".doc", ".docx"]:
        return load_word(path)
    else:
        raise ValueError("❌ 只支持 txt、pdf 和 Word（.doc/.docx）格式的知识库文件。")


def build_vector_store(docs, model):
    """
    使用句向量模型，将文本列表编码成向量
    并构建faiss的向量索引库，方便后续相似度搜索
    """
    if not docs:
        raise ValueError("❌ 文档为空，无法构建向量库。")
    embeddings = model.encode(docs, convert_to_numpy=True)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("❌ 向量生成失败，维度不合法。")
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2距离索引
    index.add(embeddings)  # 将向量加入索引
    return index, embeddings, docs


def retrieve_context(query, model, index, docs, top_k=2):
    """
    对输入query进行编码，使用faiss索引检索top_k条最相似的知识内容
    返回这些内容的拼接字符串，作为上下文提供给大模型
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)  # 返回距离和索引
    return "\n".join([docs[i] for i in I[0]])


def build_rag_prompt(query, context):
    """
    构造RAG对话消息列表，传给聊天模型
    system角色设定助手身份，user角色提供上下文和问题
    """
    return [
        {"role": "system", "content": "你是一个基于知识库的智能问答助手"},
        {"role": "user", "content": f"以下是一些知识片段：\n{context}\n\n我的问题是：{query}"}
    ]


if __name__ == "__main__":
    # 请修改为你自己的文件路径，支持txt/pdf/docx格式
    doc_path = r"E:\低空经济语料集\低空经济案例\2024年中国低空经济报告.pdf"
    # doc_path = r"D:\python-learn\rag知识库\knowledge.txt"
    # doc_path = r"D:\python-learn\rag知识库\knowledge.docx"

    # 加载知识库文档
    docs = load_documents(doc_path)

    # 加载句向量模型
    emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 构建向量索引库
    index, embeddings, docs = build_vector_store(docs, emb_model)

    print(f"✅ 向量库构建完成，知识条数：{len(docs)}")

    # 交互问答循环
    while True:
        query = input("\n请输入你的问题（输入 q 退出）：")
        if query.strip().lower() == "q":
            print("退出程序。")
            break

        # 从向量库检索相关知识上下文
        context = retrieve_context(query, emb_model, index, docs, top_k=3)

        # 构造聊天模型消息
        messages = build_rag_prompt(query, context)

        # 调用统一接口获取回答
        response = chat_with_model(messages, model_name="gpt")

        print("\n答复：", response)
