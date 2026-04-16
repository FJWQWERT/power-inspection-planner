"""
tasks_expanded_200.json 数据验证与统计脚本
检查格式完整性，统计机器人类型分布、任务类型分布，输出统计报告。
"""

import json
import os
import sys
import io
from collections import Counter

# 强制 stdout 使用 UTF-8 编码，避免 Windows GBK 编码报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def load_data(filepath):
    """加载 JSON 数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def check_format_integrity(tasks):
    """检查格式完整性，返回 (通过数, 问题列表)"""
    required_fields = ["task_id", "task_description", "robot_type"]
    recommended_fields = [
        "task_name", "action_sequence", "constraints",
        "safety_rules", "weather", "time_period", "tower", "source"
    ]
    composite_fields = ["subtasks"]
    fault_fields = ["fault_type", "fault_recovery"]

    issues = []
    passed = 0

    # 检查是否为列表
    if not isinstance(tasks, list):
        issues.append("[严重] 数据根节点不是数组")
        return 0, issues

    # 检查 task_id 唯一性
    ids = [t.get("task_id") for t in tasks]
    dup_ids = [tid for tid, cnt in Counter(ids).items() if cnt > 1]
    if dup_ids:
        issues.append(f"[严重] task_id 重复: {dup_ids}")

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"索引{i}")
        task_issues = []

        # 必填字段检查
        for field in required_fields:
            if field not in task:
                task_issues.append(f"缺少必填字段 '{field}'")
            elif not task[field]:
                task_issues.append(f"必填字段 '{field}' 为空")

        # 推荐字段检查
        missing_recommended = [f for f in recommended_fields if f not in task]
        if missing_recommended:
            task_issues.append(f"缺少推荐字段: {missing_recommended}")

        # action_sequence 类型检查
        if "action_sequence" in task:
            if not isinstance(task["action_sequence"], list):
                task_issues.append("action_sequence 应为数组")
            elif len(task["action_sequence"]) == 0:
                task_issues.append("action_sequence 为空数组")

        # 复合任务检查
        source = task.get("source", "")
        if source == "composite":
            for field in composite_fields:
                if field not in task:
                    task_issues.append(f"复合任务缺少 '{field}' 字段")
                elif not isinstance(task[field], list) or len(task[field]) < 2:
                    task_issues.append(f"复合任务 '{field}' 应至少包含2个子任务")

        # 故障注入任务检查
        if source == "fault_injection":
            for field in fault_fields:
                if field not in task:
                    task_issues.append(f"故障注入任务缺少 '{field}' 字段")
                elif not task[field]:
                    task_issues.append(f"故障注入任务 '{field}' 为空")

        if task_issues:
            issues.append(f"  [{task_id}] " + "; ".join(task_issues))
        else:
            passed += 1

    return passed, issues


def generate_report(tasks):
    """生成统计报告"""
    total = len(tasks)

    # 格式完整性检查
    passed, issues = check_format_integrity(tasks)

    # 机器人类型分布
    robot_dist = Counter(t.get("robot_type", "未知") for t in tasks)

    # 任务来源分布（param_variant / composite / fault_injection）
    source_dist = Counter(t.get("source", "未知") for t in tasks)

    # 天气分布
    weather_dist = Counter(t.get("weather", "未设置") for t in tasks)

    # 时间段分布
    time_dist = Counter(t.get("time_period", "未设置") for t in tasks)

    # 塔号分布 (top 10)
    tower_dist = Counter(t.get("tower", "未设置") for t in tasks)

    # 故障类型分布
    fault_dist = Counter(
        t["fault_type"] for t in tasks if "fault_type" in t
    )

    # 复合任务子任务数统计
    composite_tasks = [t for t in tasks if t.get("source") == "composite"]
    subtask_counts = [
        len(t.get("subtasks", [])) for t in composite_tasks
    ]

    # 动作序列长度统计
    seq_lengths = [
        len(t.get("action_sequence", [])) for t in tasks
    ]

    # ========== 输出报告 ==========
    sep = "=" * 60
    print(sep)
    print("     电力巡检任务数据集 - 验证与统计报告")
    print(sep)

    # 1. 基本信息
    print(f"\n{'─' * 40}")
    print("【1. 基本信息】")
    print(f"  任务总数:           {total}")
    print(f"  数据格式:           JSON 数组")
    print(f"  根节点类型:         {'list (正确)' if isinstance(tasks, list) else '异常'}")

    # 2. 格式完整性
    print(f"\n{'─' * 40}")
    print("【2. 格式完整性检查】")
    print(f"  检查通过:           {passed}/{total}")
    print(f"  检查状态:           {'✓ 全部通过' if passed == total else '✗ 存在问题'}")
    if issues:
        print(f"  问题列表 ({len(issues)} 项):")
        for issue in issues[:20]:  # 最多显示20条
            print(f"    {issue}")
        if len(issues) > 20:
            print(f"    ... 还有 {len(issues) - 20} 项问题")

    # 3. 机器人类型分布
    print(f"\n{'─' * 40}")
    print("【3. 机器人类型分布】")
    for rtype, count in robot_dist.most_common():
        bar = "█" * (count // 2)
        pct = count / total * 100
        print(f"  {rtype:8s}  {count:4d} 条  ({pct:5.1f}%)  {bar}")

    # 4. 任务来源分布
    print(f"\n{'─' * 40}")
    print("【4. 任务来源/类型分布】")
    source_labels = {
        "param_variant": "参数变体",
        "composite": "任务组合",
        "fault_injection": "故障注入",
    }
    for source, count in source_dist.most_common():
        label = source_labels.get(source, source)
        bar = "█" * (count // 2)
        pct = count / total * 100
        print(f"  {label:10s}  {count:4d} 条  ({pct:5.1f}%)  {bar}")

    # 5. 天气条件分布
    print(f"\n{'─' * 40}")
    print("【5. 天气条件分布】")
    for weather, count in weather_dist.most_common():
        pct = count / total * 100
        print(f"  {weather:6s}  {count:4d} 条  ({pct:5.1f}%)")

    # 6. 时间段分布
    print(f"\n{'─' * 40}")
    print("【6. 时间段分布】")
    for tp, count in time_dist.most_common():
        pct = count / total * 100
        print(f"  {tp:6s}  {count:4d} 条  ({pct:5.1f}%)")

    # 7. 塔号覆盖统计
    print(f"\n{'─' * 40}")
    print("【7. 塔号覆盖统计】")
    print(f"  覆盖塔号数:         {len(tower_dist)}")
    top_towers = tower_dist.most_common(5)
    bottom_towers = tower_dist.most_common()[-5:]
    print(f"  出现最多 (Top 5):   {', '.join(f'{t}({c}次)' for t, c in top_towers)}")
    print(f"  出现最少 (Bottom 5): {', '.join(f'{t}({c}次)' for t, c in bottom_towers)}")

    # 8. 故障类型分布
    print(f"\n{'─' * 40}")
    print("【8. 故障类型分布】")
    if fault_dist:
        for ft, count in fault_dist.most_common():
            print(f"  {ft:8s}  {count:4d} 条")
    else:
        print("  无故障注入任务")

    # 9. 复合任务统计
    print(f"\n{'─' * 40}")
    print("【9. 复合任务统计】")
    print(f"  复合任务总数:       {len(composite_tasks)}")
    if subtask_counts:
        print(f"  子任务数范围:       {min(subtask_counts)} ~ {max(subtask_counts)}")
        print(f"  平均子任务数:       {sum(subtask_counts)/len(subtask_counts):.1f}")

    # 10. 动作序列统计
    print(f"\n{'─' * 40}")
    print("【10. 动作序列统计】")
    if seq_lengths:
        print(f"  最短序列:           {min(seq_lengths)} 步")
        print(f"  最长序列:           {max(seq_lengths)} 步")
        print(f"  平均序列长度:       {sum(seq_lengths)/len(seq_lengths):.1f} 步")

    # 结论
    print(f"\n{sep}")
    if passed == total and not issues:
        print("  结论: 数据集格式完整，共 200 条任务，验证全部通过。")
    else:
        print(f"  结论: 数据集存在 {len(issues)} 个问题，请检查修复。")
    print(sep)

    return passed == total and not issues


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    filepath = os.path.join(project_dir, "data", "tasks_expanded_200.json")

    if not os.path.exists(filepath):
        print(f"[错误] 文件不存在: {filepath}")
        sys.exit(1)

    tasks = load_data(filepath)
    success = generate_report(tasks)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
