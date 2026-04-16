"""
TaskPlanner 测试脚本
测试3条不同巡检指令的规划结果。
"""

import json
import os
import sys
import io

# 强制 stdout 使用 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 使用 HuggingFace 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 将项目根目录加入 sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.planner.task_planner import TaskPlanner


def print_result(index, instruction, result):
    """格式化打印单次规划结果"""
    sep = "=" * 60
    thin = "─" * 60

    print(f"\n{sep}")
    print(f"  测试 {index}: {instruction}")
    print(sep)

    if result.get("success"):
        print(f"\n  [状态] 规划成功")

        # 任务序列
        tasks = result.get("task_sequence", [])
        print(f"  [任务数量] {len(tasks)}")
        for t in tasks:
            step = t.get("step", "?")
            name = t.get("task_name", "未知")
            robot = t.get("robot_type", "未知")
            desc = t.get("task_description", "")
            print(f"\n    步骤 {step}: {name}")
            print(f"      机器人: {robot}")
            print(f"      描述:   {desc}")
            actions = t.get("action_sequence", [])
            if actions:
                print(f"      动作序列:")
                for a in actions:
                    print(f"        - {a}")
            constraints = t.get("constraints", [])
            if constraints:
                print(f"      约束: {', '.join(constraints)}")
            safety = t.get("safety_rules", [])
            if safety:
                print(f"      安全: {', '.join(safety)}")

        # 机器人分配
        assignment = result.get("robot_assignment", {})
        print(f"\n  [机器人分配]")
        for robot, task_list in assignment.items():
            if task_list:
                print(f"    {robot}: {', '.join(task_list)}")
            else:
                print(f"    {robot}: (无任务)")

        # 预估时间
        print(f"  [预估时间] {result.get('estimated_time', '未知')} 分钟")

        # 补充说明
        notes = result.get("notes")
        if notes:
            print(f"  [备注] {notes}")

        # 完整 JSON 输出
        print(f"\n  {thin}")
        print(f"  完整 JSON 输出:")
        print(f"  {thin}")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        print(f"\n  [状态] 规划失败")
        print(f"  [错误] {result.get('error', '未知错误')}")
        raw = result.get("raw_response")
        if raw:
            print(f"\n  LLM 原始回复 (前500字):")
            print(f"  {raw[:500]}")

    print(f"\n{sep}\n")


def main():
    banner = "=" * 60
    print(banner)
    print("  TaskPlanner 测试脚本")
    print(f"  共 3 条测试指令")
    print(banner)

    # 初始化规划器（只初始化一次，复用模型和知识库）
    planner = TaskPlanner()

    # 3条测试指令
    test_instructions = [
        "对#12塔绝缘子进行红外测温",
        "巡视#23到#25塔之间的导线",
        "先对#12塔测温，再巡视#23-#25塔",
    ]

    results = []
    for i, instruction in enumerate(test_instructions, 1):
        print(f"\n{'*' * 60}")
        print(f"  正在执行测试 {i}/{len(test_instructions)}...")
        print(f"{'*' * 60}")

        result = planner.plan(instruction)
        results.append(result)
        print_result(i, instruction, result)

    # 汇总
    print(f"\n{banner}")
    print("  测试汇总")
    print(banner)
    for i, (inst, res) in enumerate(zip(test_instructions, results), 1):
        status = "成功" if res.get("success") else "失败"
        task_count = len(res.get("task_sequence", []))
        est_time = res.get("estimated_time", "?")
        print(f"  {i}. [{status}] {inst}")
        print(f"     任务数: {task_count}, 预估时间: {est_time} 分钟")
    print(banner)


if __name__ == "__main__":
    main()
