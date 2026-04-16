"""
电力巡检任务规划器
整合 RAG 知识库检索 + LLM 调用，根据用户指令生成巡检任务规划方案。
"""

import json
import os
import sys
import io
import re

# 强制 stdout 使用 UTF-8 编码（仅在未设置时执行，避免重复包装）
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 使用 HuggingFace 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 将项目根目录加入 sys.path，确保模块可导入
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.api.llm_client import call_llm

# 系统提示词
SYSTEM_MESSAGE = """你是电力巡检任务规划专家。你需要根据用户的巡检指令，结合已有的任务知识库，生成合理的巡检任务规划方案。

可用机器人类型：
- UAV（无人机）：适合高空巡检、线路巡视、红外测温、通道巡查等空中作业
- WHEEL（轮式机器人）：适合变电站内部巡检、仪表读数、设备状态检查等地面作业
- ARM（机械臂机器人）：适合开关操作、设备更换、传感器安装等精细操作

请严格按照以下JSON格式输出，不要输出任何其他内容：
{
    "task_sequence": [
        {
            "step": 1,
            "task_name": "任务名称",
            "task_description": "详细描述",
            "robot_type": "UAV/WHEEL/ARM",
            "action_sequence": ["步骤1", "步骤2", "..."],
            "constraints": ["约束条件1", "..."],
            "safety_rules": ["安全规则1", "..."]
        }
    ],
    "robot_assignment": {
        "UAV": ["任务名称1"],
        "WHEEL": ["任务名称2"],
        "ARM": ["任务名称3"]
    },
    "estimated_time": "预估总时间（分钟）",
    "notes": "补充说明"
}"""


class TaskPlanner:
    """电力巡检任务规划器"""

    def __init__(self, kb_path=None, tasks_path=None):
        """
        初始化任务规划器

        参数:
            kb_path:    知识库路径，默认为项目下 kb 目录
            tasks_path: 原始任务数据路径，默认为 data/tasks_original.json
        """
        self.project_dir = PROJECT_DIR
        self.kb_path = kb_path or os.path.join(self.project_dir, "kb")
        self.tasks_path = tasks_path or os.path.join(
            self.project_dir, "data", "tasks_original.json"
        )

        print("=" * 60)
        print("  TaskPlanner 初始化")
        print("=" * 60)

        # 加载原始任务数据
        print("\n[初始化 1/3] 加载原始任务数据...")
        self.tasks = self._load_tasks()
        print(f"  已加载 {len(self.tasks)} 条任务")

        # 加载 Embedding 模型
        print("\n[初始化 2/3] 加载 Embedding 模型 (BAAI/bge-small-zh)...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("  模型加载完成")

        # 加载知识库
        print("\n[初始化 3/3] 加载 Chroma 知识库...")
        self.vectorstore = Chroma(
            persist_directory=self.kb_path,
            embedding_function=self.embedding_model,
            collection_name="power_inspection_tasks",
        )
        doc_count = self.vectorstore._collection.count()
        print(f"  知识库加载完成，共 {doc_count} 条文档")

        print("\n" + "=" * 60)
        print("  TaskPlanner 初始化完成，准备就绪!")
        print("=" * 60)

    def _load_tasks(self):
        """加载原始任务数据"""
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _retrieve_similar_tasks(self, query, top_k=5):
        """从知识库检索最相似的任务"""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return results

    def _build_prompt(self, user_instruction, similar_tasks):
        """构建发送给 LLM 的 prompt"""
        # 格式化检索到的相似任务
        reference_text = ""
        for i, (doc, score) in enumerate(similar_tasks, 1):
            meta = doc.metadata
            reference_text += (
                f"\n参考任务 {i} (相似度: {score:.4f}):\n"
                f"  编号: {meta['task_id']}\n"
                f"  名称: {meta['task_name']}\n"
                f"  机器人: {meta['robot_type']}\n"
                f"  描述: {meta['task_description']}\n"
            )

        prompt = f"""## 知识库检索到的参考任务
{reference_text}

## 用户巡检指令
{user_instruction}

请根据以上参考任务和用户指令，生成完整的巡检任务规划方案。严格按照JSON格式输出。"""

        return prompt

    def _parse_llm_response(self, response_text):
        """解析 LLM 返回的 JSON"""
        # 尝试从回复中提取 JSON 块
        # 优先匹配 ```json ... ``` 代码块
        json_match = re.search(r"```json\s*\n?(.*?)\n?\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # 尝试匹配 ``` ... ``` 代码块
            json_match = re.search(r"```\s*\n?(.*?)\n?\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # 尝试直接匹配 { ... } 最外层
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text.strip()

        return json.loads(json_str)

    def plan(self, user_instruction):
        """
        根据用户指令生成巡检任务规划

        参数:
            user_instruction: 用户的巡检指令文本

        返回:
            dict: 包含 task_sequence、robot_assignment、estimated_time 的规划方案
        """
        sep = "─" * 50

        print(f"\n{sep}")
        print(f"  开始规划任务...")
        print(f"  用户指令: {user_instruction}")
        print(sep)

        # 步骤1: 知识库检索
        print("\n[步骤 1/4] 从知识库检索相似任务...")
        similar_tasks = self._retrieve_similar_tasks(user_instruction, top_k=5)
        print(f"  检索到 {len(similar_tasks)} 个相似任务:")
        for i, (doc, score) in enumerate(similar_tasks, 1):
            print(f"    {i}. [{doc.metadata['task_id']}] "
                  f"{doc.metadata['task_name']} "
                  f"(得分: {score:.4f})")

        # 步骤2: 构建 prompt
        print("\n[步骤 2/4] 构建 LLM Prompt...")
        prompt = self._build_prompt(user_instruction, similar_tasks)
        print(f"  Prompt 长度: {len(prompt)} 字符")
        print(f"  系统消息: {SYSTEM_MESSAGE[:50]}...")

        # 步骤3: 调用 LLM
        print("\n[步骤 3/4] 调用 LLM (DeepSeek-V3)...")
        try:
            response_text = call_llm(
                prompt=prompt,
                system_message=SYSTEM_MESSAGE,
                temperature=0.3,
                max_tokens=3000,
            )
            print(f"  LLM 返回内容长度: {len(response_text)} 字符")
        except Exception as e:
            print(f"  LLM 调用失败: {e}")
            return {
                "success": False,
                "error": f"LLM 调用失败: {str(e)}",
                "task_sequence": [],
                "robot_assignment": {},
                "estimated_time": "未知",
            }

        # 步骤4: 解析结果
        print("\n[步骤 4/4] 解析 LLM 返回结果...")
        try:
            result = self._parse_llm_response(response_text)
            result["success"] = True

            # 统计信息
            task_count = len(result.get("task_sequence", []))
            robots_used = list(result.get("robot_assignment", {}).keys())
            est_time = result.get("estimated_time", "未知")

            print(f"  解析成功!")
            print(f"  任务数量: {task_count}")
            print(f"  使用机器人: {', '.join(robots_used)}")
            print(f"  预估时间: {est_time}")

            return result

        except json.JSONDecodeError as e:
            print(f"  JSON 解析失败: {e}")
            print(f"  LLM 原始回复:")
            for line in response_text.split("\n")[:10]:
                print(f"    {line}")
            return {
                "success": False,
                "error": f"LLM 返回内容无法解析为JSON: {str(e)}",
                "raw_response": response_text,
                "task_sequence": [],
                "robot_assignment": {},
                "estimated_time": "未知",
            }

        except Exception as e:
            print(f"  结果解析异常: {e}")
            return {
                "success": False,
                "error": f"结果处理异常: {str(e)}",
                "raw_response": response_text,
                "task_sequence": [],
                "robot_assignment": {},
                "estimated_time": "未知",
            }


# ============ 测试代码 ============
if __name__ == "__main__":
    # 初始化规划器
    planner = TaskPlanner()

    # 测试规划
    test_instruction = "请规划一次对#12塔的全面巡检，包括绝缘子红外测温、导线弧垂检查和杆塔倾斜检测"

    result = planner.plan(test_instruction)

    # 输出结果
    sep = "=" * 60
    print(f"\n{sep}")
    print("  规划结果")
    print(sep)

    if result.get("success"):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\n  规划失败: {result.get('error')}")
        if "raw_response" in result:
            print(f"\n  LLM 原始回复:\n  {result['raw_response'][:500]}")

    print(sep)
