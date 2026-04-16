"""
电力巡检任务数据扩充脚本
从 data/tasks_original.json 读取20条基础任务，通过参数变体、任务组合、故障注入三种方式扩充到200条。
输出到 data/tasks_expanded_200.json
"""

import json
import random
import copy
import re
import os

# ============ 配置 ============
TOWERS = [f"#{i:02d}" for i in range(1, 31)]          # #01 ~ #30
WEATHERS = ["晴天", "阴天", "小雨", "雾天"]
TIME_PERIODS = ["白天", "夜间"]

FAULT_TYPES = [
    {
        "fault_type": "通信中断",
        "fault_recovery": "切换至备用通信频段，若仍无法恢复则自动返航并记录断点位置"
    },
    {
        "fault_type": "电量不足",
        "fault_recovery": "立即中止当前任务，以最短路径返回充电桩/起降点，保存已采集数据"
    },
    {
        "fault_type": "识别失败",
        "fault_recovery": "调整拍摄角度和距离重试3次，仍失败则标记为人工复核并继续下一目标"
    },
    {
        "fault_type": "路径受阻",
        "fault_recovery": "启动避障模块重新规划路径，若无可用路径则原路返回并上报障碍物信息"
    }
]

random.seed(42)


def load_original_tasks(filepath):
    """加载原始任务数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def replace_tower_in_text(text, new_tower):
    """将文本中的塔号/柜号/设备编号替换为新编号"""
    # 替换 #数字 格式
    return re.sub(r"#\d+", new_tower, text)


def generate_param_variants(tasks, target_count):
    """
    参数变体扩充：为每个原始任务随机生成变体
    变化维度：塔号、天气、时间段
    """
    variants = []
    task_idx = 0

    while len(variants) < target_count:
        base = tasks[task_idx % len(tasks)]
        tower = random.choice(TOWERS)
        weather = random.choice(WEATHERS)
        time_period = random.choice(TIME_PERIODS)

        variant = copy.deepcopy(base)
        seq_num = len(variants) + 1
        prefix = base["robot_type"]
        variant["task_id"] = f"{prefix}-V{seq_num:03d}"

        # 替换描述中的塔号
        variant["task_description"] = replace_tower_in_text(
            base["task_description"], tower
        )

        # 添加环境参数
        variant["weather"] = weather
        variant["time_period"] = time_period
        variant["tower"] = tower
        variant["source"] = "param_variant"
        variant["base_task_id"] = base["task_id"]

        # 根据天气和时间调整约束
        extra_constraints = []
        if weather == "小雨":
            extra_constraints.append("注意防水防滑")
        if weather == "雾天":
            extra_constraints.append("能见度不足时暂停作业")
        if time_period == "夜间":
            extra_constraints.append("开启夜间照明设备")

        variant["constraints"] = base["constraints"] + extra_constraints

        # 根据天气调整安全规则
        extra_safety = []
        if weather in ("小雨", "雾天"):
            extra_safety.append("恶劣天气加强通信监控")
        if time_period == "夜间":
            extra_safety.append("夜间作业需有值班人员在线监控")

        variant["safety_rules"] = base["safety_rules"] + extra_safety

        variants.append(variant)
        task_idx += 1

    return variants


def generate_composite_tasks(tasks, target_count):
    """
    任务组合扩充：将基础任务两两组合成复合任务
    复合任务包含 subtasks 字段
    """
    composites = []
    # 按 robot_type 分组，只组合相同类型的任务
    by_type = {}
    for t in tasks:
        by_type.setdefault(t["robot_type"], []).append(t)

    pairs = []
    for rtype, group in by_type.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.append((group[i], group[j]))

    random.shuffle(pairs)

    for idx, (t1, t2) in enumerate(pairs):
        if len(composites) >= target_count:
            break

        seq_num = idx + 1
        prefix = t1["robot_type"]
        composite = {
            "task_id": f"{prefix}-C{seq_num:03d}",
            "task_name": f"{t1['task_name']}+{t2['task_name']}",
            "robot_type": t1["robot_type"],
            "task_description": f"复合任务：先执行{t1['task_name']}，再执行{t2['task_name']}",
            "subtasks": [
                {
                    "subtask_id": t1["task_id"],
                    "subtask_name": t1["task_name"],
                    "task_description": t1["task_description"],
                    "action_sequence": t1["action_sequence"]
                },
                {
                    "subtask_id": t2["task_id"],
                    "subtask_name": t2["task_name"],
                    "task_description": t2["task_description"],
                    "action_sequence": t2["action_sequence"]
                }
            ],
            "action_sequence": t1["action_sequence"][:-1] + t2["action_sequence"],
            "constraints": list(set(t1["constraints"] + t2["constraints"])),
            "safety_rules": list(set(t1["safety_rules"] + t2["safety_rules"])),
            "weather": random.choice(WEATHERS),
            "time_period": random.choice(TIME_PERIODS),
            "tower": random.choice(TOWERS),
            "source": "composite"
        }
        composites.append(composite)

    return composites


def generate_fault_tasks(tasks, target_count):
    """
    故障注入扩充：为部分任务添加故障场景
    添加 fault_type 和 fault_recovery 字段
    """
    fault_tasks = []

    for idx in range(target_count):
        base = tasks[idx % len(tasks)]
        fault = random.choice(FAULT_TYPES)

        variant = copy.deepcopy(base)
        seq_num = idx + 1
        prefix = base["robot_type"]
        variant["task_id"] = f"{prefix}-F{seq_num:03d}"
        variant["task_name"] = f"{base['task_name']}（{fault['fault_type']}场景）"
        variant["task_description"] = (
            f"{base['task_description']}（模拟{fault['fault_type']}故障场景）"
        )
        variant["fault_type"] = fault["fault_type"]
        variant["fault_recovery"] = fault["fault_recovery"]
        variant["weather"] = random.choice(WEATHERS)
        variant["time_period"] = random.choice(TIME_PERIODS)
        variant["tower"] = random.choice(TOWERS)
        variant["source"] = "fault_injection"
        variant["base_task_id"] = base["task_id"]

        # 在动作序列中插入故障处理步骤
        seq = copy.deepcopy(base["action_sequence"])
        if len(seq) > 2:
            insert_pos = random.randint(1, len(seq) - 1)
            seq.insert(insert_pos, f"【故障触发】{fault['fault_type']}")
            seq.insert(insert_pos + 1, f"【故障恢复】{fault['fault_recovery']}")
        variant["action_sequence"] = seq

        fault_tasks.append(variant)

    return fault_tasks


def validate_task(task):
    """验证每个任务都包含必要字段"""
    required = ["task_id", "task_description", "robot_type"]
    for field in required:
        if field not in task or not task[field]:
            raise ValueError(f"任务 {task.get('task_id', 'UNKNOWN')} 缺少字段: {field}")


def main():
    # 确定路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_path = os.path.join(project_dir, "data", "tasks_original.json")
    output_path = os.path.join(project_dir, "data", "tasks_expanded_200.json")

    # 1. 加载原始数据
    original_tasks = load_original_tasks(input_path)
    print(f"[1/4] 已加载 {len(original_tasks)} 条原始任务")

    # 2. 参数变体扩充 —— 生成 100 条
    param_variants = generate_param_variants(original_tasks, 100)
    print(f"[2/4] 参数变体扩充: 生成 {len(param_variants)} 条")

    # 3. 任务组合扩充 —— 生成 60 条
    composite_tasks = generate_composite_tasks(original_tasks, 60)
    print(f"[3/4] 任务组合扩充: 生成 {len(composite_tasks)} 条")

    # 4. 故障注入扩充 —— 生成 40 条
    fault_tasks = generate_fault_tasks(original_tasks, 40)
    print(f"[4/4] 故障注入扩充: 生成 {len(fault_tasks)} 条")

    # 合并所有任务
    all_tasks = param_variants + composite_tasks + fault_tasks
    print(f"\n合并后总数: {len(all_tasks)} 条")

    # 如果超过200条，截取前200条
    if len(all_tasks) > 200:
        all_tasks = all_tasks[:200]
        print(f"截取至 200 条")

    # 如果不足200条，从参数变体中补充
    while len(all_tasks) < 200:
        extra = generate_param_variants(original_tasks, 200 - len(all_tasks))
        for i, t in enumerate(extra):
            t["task_id"] = f"{t['robot_type']}-E{len(all_tasks)+i+1:03d}"
        all_tasks.extend(extra)
    all_tasks = all_tasks[:200]

    # 重新编号确保 task_id 唯一
    for i, task in enumerate(all_tasks):
        task["task_id"] = f"TASK-{i+1:03d}"

    # 验证所有任务
    for task in all_tasks:
        validate_task(task)
    print(f"验证通过: 所有 {len(all_tasks)} 条任务包含必要字段")

    # 统计信息
    sources = {}
    for t in all_tasks:
        s = t.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    print(f"\n数据来源分布:")
    for s, c in sources.items():
        print(f"  - {s}: {c} 条")

    robot_types = {}
    for t in all_tasks:
        rt = t["robot_type"]
        robot_types[rt] = robot_types.get(rt, 0) + 1
    print(f"\n机器人类型分布:")
    for rt, c in robot_types.items():
        print(f"  - {rt}: {c} 条")

    # 输出
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, ensure_ascii=False, indent=2)

    print(f"\n已输出到: {output_path}")
    print(f"最终任务数量: {len(all_tasks)} 条")


if __name__ == "__main__":
    main()
