"""
电力巡检任务规划系统 - Flask Web 应用
访问 http://127.0.0.1:5000
"""

import json
import os
import sys
import io
import time
import subprocess
from pathlib import Path

# 强制 stdout 使用 UTF-8 编码
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 使用 HuggingFace 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 将项目根目录加入 sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from flask import Flask, render_template, request, jsonify
from src.planner.task_planner import TaskPlanner

app = Flask(__name__)

# Webots 联动：共享目录
SHARED_DIR = Path(__file__).parent / 'shared'
TASK_FILE = SHARED_DIR / 'task.json'       # 写入任务给机器人
RESULT_FILE = SHARED_DIR / 'result.json'   # 读取机器人结果
STATUS_FILE = SHARED_DIR / 'status.json'   # 读取机器人状态
SHARED_DIR.mkdir(exist_ok=True)

# Webots 世界文件路径
WEBOTS_WORLD = Path(__file__).parent / 'webots' / 'worlds' / 'power_inspection.wbt'

# 全局 Webots 进程引用
webots_process = None

# 全局规划器实例（启动时初始化一次，避免重复加载模型）
planner = None


def get_planner():
    """懒加载 TaskPlanner 单例"""
    global planner
    if planner is None:
        print("\n[Flask] 初始化 TaskPlanner...")
        planner = TaskPlanner()
        print("[Flask] TaskPlanner 初始化完成\n")
    return planner


def send_task_to_robot(action, target_position, target="", defect_type=""):
    """将巡检任务写入 task.json，格式与 patrol_controller 期望的一致。

    参数:
        action:          动作类型，如 "inspect"
        target_position: 目标坐标 [x, y, z]，控制器取 x/z 作为地面导航坐标
        target:          目标名称，如 "insulator_1"（可选）
        defect_type:     缺陷类型，如 "破损"（可选）

    返回:
        dict: 写入的完整任务数据
    """
    task = {
        "task_id": int(time.time()),
        "target": target,
        "target_position": target_position,
        "action": action,
        "defect_type": defect_type,
        "dispatched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    TASK_FILE.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[send_task_to_robot] {action} -> {target_position}")
    return task


@app.route("/")
def index():
    """首页"""
    return render_template("index.html")


@app.route("/plan", methods=["POST"])
def plan():
    """接收指令，调用规划器，返回结果"""
    data = request.get_json()
    instruction = data.get("instruction", "").strip()

    if not instruction:
        return jsonify({"success": False, "error": "请输入巡检指令"})

    try:
        p = get_planner()
        result = p.plan(instruction)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"规划过程出错: {str(e)}",
            "task_sequence": [],
            "robot_assignment": {},
            "estimated_time": "未知",
        })


# ============ Webots 联动接口 ============

# 预定义的巡检目标位置（与 Webots 世界文件中的物体对应）
INSPECTION_TARGETS = {
    "insulator_1": [3, 0, 5],
    "insulator_2": [-4, 0, 3],
    "insulator_3": [5, 0, -4],
}


@app.route("/dispatch", methods=["POST"])
def dispatch_task():
    """将巡检任务下发给 Webots 机器人（写入 task.json）"""
    data = request.get_json()
    target = data.get("target", "insulator_1")
    defect_type = data.get("defect_type", "")
    action = data.get("action", "inspect")

    target_position = INSPECTION_TARGETS.get(target)
    if not target_position:
        target_position = data.get("target_position", [0, 0, 0])

    task = {
        "task_id": int(time.time()),
        "target": target,
        "target_position": target_position,
        "action": action,
        "defect_type": defect_type,
        "dispatched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with open(TASK_FILE, "w", encoding="utf-8") as f:
            json.dump(task, f, ensure_ascii=False, indent=2)
        print(f"[Flask] 任务已下发: {target} -> {target_position}")
        return jsonify({"success": True, "task": task})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/robot_result", methods=["GET"])
def robot_result():
    """读取 Webots 机器人的巡检结果"""
    if not RESULT_FILE.exists():
        return jsonify({"success": False, "status": "waiting",
                        "message": "暂无结果，机器人可能仍在执行任务"})
    try:
        with open(RESULT_FILE, "r", encoding="utf-8") as f:
            result = json.load(f)
        return jsonify({"success": True, "result": result})
    except (json.JSONDecodeError, IOError) as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/robot_result", methods=["DELETE"])
def clear_robot_result():
    """清除上一次的巡检结果"""
    if RESULT_FILE.exists():
        RESULT_FILE.unlink()
    return jsonify({"success": True})


@app.route("/targets", methods=["GET"])
def get_targets():
    """获取可用的巡检目标列表"""
    targets = [
        {"id": k, "name": k.replace("_", " ").title(), "position": v}
        for k, v in INSPECTION_TARGETS.items()
    ]
    return jsonify({"success": True, "targets": targets})


# ============ 新增 /api/ 接口 ============

@app.route("/api/send_task", methods=["POST"])
def api_send_task():
    """发送巡检任务到 task.json，供 Webots 控制器读取"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求体不能为空"}), 400

    target = data.get("target", "")
    target_position = data.get("target_position")
    action = data.get("action", "inspect")
    defect_type = data.get("defect_type", "")

    # 如果没有提供坐标，尝试从预定义目标中查找
    if not target_position:
        target_position = INSPECTION_TARGETS.get(target)
    if not target_position:
        return jsonify({"success": False,
                        "error": f"未知目标 '{target}'，请提供 target_position 或使用预定义目标: {list(INSPECTION_TARGETS.keys())}"}), 400

    task = {
        "task_id": int(time.time()),
        "target": target,
        "target_position": target_position,
        "action": action,
        "defect_type": defect_type,
        "dispatched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        TASK_FILE.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Flask] /api/send_task -> {target} {target_position}")
        return jsonify({"success": True, "task": task})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/robot_result", methods=["GET"])
def api_robot_result():
    """读取 result.json 中 Webots 机器人的巡检结果"""
    if not RESULT_FILE.exists():
        return jsonify({"success": False, "status": "waiting",
                        "message": "暂无巡检结果，机器人可能仍在执行任务"})
    try:
        result = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
        return jsonify({"success": True, "result": result})
    except (json.JSONDecodeError, IOError) as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/robot_status", methods=["GET"])
def api_robot_status():
    """获取机器人当前状态（综合 task/result/status 文件判断）"""
    status_info = {
        "webots_running": webots_process is not None and webots_process.poll() is None,
        "state": "unknown",
        "current_task": None,
        "last_result": None,
    }

    # 如果 status.json 存在（控制器主动上报的状态）
    if STATUS_FILE.exists():
        try:
            status_info.update(json.loads(STATUS_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, IOError):
            pass

    # 通过 task.json / result.json 推断状态
    if TASK_FILE.exists():
        try:
            status_info["current_task"] = json.loads(TASK_FILE.read_text(encoding="utf-8"))
            status_info["state"] = "navigating"
        except (json.JSONDecodeError, IOError):
            pass
    elif RESULT_FILE.exists():
        try:
            status_info["last_result"] = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
            status_info["state"] = "idle"
        except (json.JSONDecodeError, IOError):
            pass
    else:
        status_info["state"] = "idle"

    return jsonify({"success": True, "status": status_info})


@app.route("/api/start_webots", methods=["POST"])
def api_start_webots():
    """启动 Webots 仿真（打开世界文件）"""
    global webots_process

    # 检查是否已在运行
    if webots_process is not None and webots_process.poll() is None:
        return jsonify({"success": False, "message": "Webots 仿真已在运行中",
                        "pid": webots_process.pid})

    if not WEBOTS_WORLD.exists():
        return jsonify({"success": False,
                        "error": f"世界文件不存在: {WEBOTS_WORLD}"}), 404

    # 在 PATH 和常见安装路径中查找 webots 可执行文件
    webots_exe = None
    common_paths = [
        r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe",
        r"C:\Program Files\Webots\bin\webots.exe",
        r"C:\Program Files (x86)\Webots\msys64\mingw64\bin\webots.exe",
        r"D:\Program Files\Webots\msys64\mingw64\bin\webots.exe",
        r"D:\Webots\msys64\mingw64\bin\webots.exe",
    ]

    # 优先使用环境变量指定的路径
    env_webots = os.environ.get("WEBOTS_HOME")
    if env_webots:
        candidate = Path(env_webots) / "msys64" / "mingw64" / "bin" / "webots.exe"
        if candidate.exists():
            webots_exe = str(candidate)

    if not webots_exe:
        for p in common_paths:
            if Path(p).exists():
                webots_exe = p
                break

    if not webots_exe:
        return jsonify({
            "success": False,
            "error": "未找到 Webots，请设置 WEBOTS_HOME 环境变量或将 webots 加入 PATH",
        }), 404

    try:
        webots_process = subprocess.Popen(
            [webots_exe, str(WEBOTS_WORLD)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[Flask] Webots 已启动, PID={webots_process.pid}")
        return jsonify({"success": True, "message": "Webots 仿真已启动",
                        "pid": webots_process.pid,
                        "world": str(WEBOTS_WORLD)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # 启动时预加载规划器
    get_planner()
    print("\n[Flask] 启动 Web 服务...")
    print("[Flask] 访问地址: http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
