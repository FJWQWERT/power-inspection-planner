"""
基于 LangChain 构建电力巡检任务 RAG 知识库
- 读取 data/tasks_original.json (20条任务)
- 使用 HuggingFaceEmbeddings (BAAI/bge-small-zh)
- 使用 Chroma 向量数据库
- 知识库保存到 ./kb
- 测试检索 '对绝缘子进行红外测温'
"""

import json
import os
import sys
import io
import shutil

# 强制 stdout 使用 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 使用 HuggingFace 国内镜像，解决网络访问问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def load_tasks(filepath):
    """加载原始任务数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def tasks_to_documents(tasks):
    """将任务数据转换为 LangChain Document 对象"""
    documents = []
    for task in tasks:
        # 构造检索内容：包含 task_id、task_name、robot_type、task_description
        content = (
            f"任务编号: {task['task_id']}\n"
            f"任务名称: {task['task_name']}\n"
            f"机器人类型: {task['robot_type']}\n"
            f"任务描述: {task['task_description']}"
        )

        # 将完整任务信息存入 metadata，便于检索后获取详细数据
        metadata = {
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "robot_type": task["robot_type"],
            "task_description": task["task_description"],
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def build_knowledge_base(documents, embedding_model, kb_path):
    """构建并持久化 Chroma 向量知识库"""
    # 如果知识库目录已存在，先清除重建
    if os.path.exists(kb_path):
        shutil.rmtree(kb_path)
        print(f"  已清除旧知识库: {kb_path}")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=kb_path,
        collection_name="power_inspection_tasks",
    )

    return vectorstore


def test_retrieval(vectorstore, query, top_k=3):
    """测试检索功能"""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return results


def main():
    # 路径配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    tasks_path = os.path.join(project_dir, "data", "tasks_expanded_200.json")
    kb_path = os.path.join(project_dir, "kb")

    sep = "=" * 60

    # ===== 1. 加载任务数据 =====
    print(sep)
    print("  电力巡检任务 RAG 知识库构建")
    print(sep)

    print("\n[1/4] 加载任务数据...")
    tasks = load_tasks(tasks_path)
    print(f"  加载完成: {len(tasks)} 条任务")

    # ===== 2. 转换为 Document =====
    print("\n[2/4] 转换为 LangChain Document...")
    documents = tasks_to_documents(tasks)
    print(f"  转换完成: {len(documents)} 个文档")
    print(f"  示例文档内容:")
    print(f"  ---")
    for line in documents[0].page_content.split("\n"):
        print(f"    {line}")
    print(f"  ---")

    # ===== 3. 加载 Embedding 模型并构建知识库 =====
    print("\n[3/4] 加载 Embedding 模型 (BAAI/bge-small-zh)...")
    print("  首次运行需要下载模型，请耐心等待...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("  模型加载完成")

    print(f"\n  构建 Chroma 向量知识库...")
    vectorstore = build_knowledge_base(documents, embedding_model, kb_path)
    print(f"  知识库已保存到: {kb_path}")
    print(f"  向量维度: {len(embedding_model.embed_query('test'))}")
    print(f"  文档数量: {vectorstore._collection.count()}")

    # ===== 4. 测试检索 =====
    print(f"\n[4/4] 测试检索...")
    query = "对绝缘子进行红外测温"
    print(f"  查询: '{query}'")
    print(f"  返回 Top 3 最相似任务:\n")

    results = test_retrieval(vectorstore, query, top_k=3)

    for i, (doc, score) in enumerate(results, 1):
        print(f"  {'─' * 50}")
        print(f"  排名 #{i}  |  相似度得分: {score:.4f}")
        print(f"  任务编号:   {doc.metadata['task_id']}")
        print(f"  任务名称:   {doc.metadata['task_name']}")
        print(f"  机器人类型: {doc.metadata['robot_type']}")
        print(f"  任务描述:   {doc.metadata['task_description']}")

    print(f"\n{sep}")
    print("  RAG 知识库构建与测试完成!")
    print(sep)


if __name__ == "__main__":
    main()
