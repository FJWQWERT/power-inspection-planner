"""
Webots Pioneer3dx 电力巡检控制器
- 监听 shared/task.json 获取巡检任务
- 控制机器人导航至目标位置
- 到达后模拟拍照巡检
- 将结果写入 shared/result.json
"""

import json
import math
import os
import time

from controller import Robot

# ============ 配置 ============

# 共享目录路径（与 Flask Web 端约定的交互目录）
# 优先使用环境变量，否则使用项目默认路径
SHARED_DIR = os.environ.get(
    "PATROL_SHARED_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"),
)
TASK_FILE = os.path.join(SHARED_DIR, "task.json")
RESULT_FILE = os.path.join(SHARED_DIR, "result.json")

# 导航参数
ARRIVE_THRESHOLD = 0.3       # 到达目标判定距离（米）
MAX_SPEED = 5.0              # 最大轮速（rad/s）
TURN_KP = 3.0                # 转向比例系数
FORWARD_KP = 2.0             # 前进比例系数
POLL_INTERVAL_STEPS = 10     # 每隔多少步检查一次任务文件
INSPECTION_DURATION = 3.0    # 模拟巡检持续时间（秒）

# ============ 工具函数 ============


def normalize_angle(angle):
    """将角度归一化到 [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_bearing(compass_values):
    """从 Compass 读数计算航向角（弧度，北为 0，顺时针为正）"""
    # Webots Compass 返回 [x, y, z]，其中 x 指北
    rad = math.atan2(compass_values[0], compass_values[2])
    return rad


def load_task():
    """尝试读取任务文件，返回任务 dict 或 None"""
    if not os.path.exists(TASK_FILE):
        return None
    try:
        with open(TASK_FILE, "r", encoding="utf-8") as f:
            task = json.load(f)
        # 验证必要字段
        if "target_position" not in task:
            return None
        return task
    except (json.JSONDecodeError, IOError):
        return None


def consume_task():
    """读取并删除任务文件（防止重复执行）"""
    task = load_task()
    if task and os.path.exists(TASK_FILE):
        try:
            os.remove(TASK_FILE)
        except OSError:
            pass
    return task


def save_result(result):
    """将巡检结果写入 result.json"""
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def clamp(value, min_val, max_val):
    """限幅"""
    return max(min_val, min(max_val, value))


# ============ 控制器状态机 ============

STATE_IDLE = "idle"              # 等待任务
STATE_NAVIGATING = "navigating"  # 导航中
STATE_INSPECTING = "inspecting"  # 巡检中（到达目标后）
STATE_DONE = "done"              # 巡检完成，写结果


class PatrolController:
    """Pioneer3dx 巡检控制器"""

    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # 初始化电机
        self.left_motor = self.robot.getDevice("left wheel")
        self.right_motor = self.robot.getDevice("right wheel")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # 初始化 GPS
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)

        # 初始化 Compass
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)

        # 初始化距离传感器（避障用）
        self.sonars = []
        for i in range(16):
            sensor = self.robot.getDevice(f"so{i}")
            sensor.enable(self.timestep)
            self.sonars.append(sensor)

        # 状态
        self.state = STATE_IDLE
        self.current_task = None
        self.target_x = 0.0
        self.target_z = 0.0
        self.inspect_start_time = 0.0
        self.step_count = 0

        print("[PatrolController] 初始化完成")
        print(f"[PatrolController] 任务文件: {TASK_FILE}")
        print(f"[PatrolController] 结果文件: {RESULT_FILE}")

    def stop(self):
        """停止机器人"""
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def set_speed(self, left, right):
        """设置左右轮速度"""
        self.left_motor.setVelocity(clamp(left, -MAX_SPEED, MAX_SPEED))
        self.right_motor.setVelocity(clamp(right, -MAX_SPEED, MAX_SPEED))

    def get_position(self):
        """获取当前 GPS 位置 (x, z)"""
        values = self.gps.getValues()
        return values[0], values[2]

    def get_heading(self):
        """获取当前航向角"""
        return get_bearing(self.compass.getValues())

    def distance_to_target(self):
        """计算到目标点的距离"""
        x, z = self.get_position()
        dx = self.target_x - x
        dz = self.target_z - z
        return math.sqrt(dx * dx + dz * dz)

    def bearing_to_target(self):
        """计算到目标点的方位角"""
        x, z = self.get_position()
        dx = self.target_x - x
        dz = self.target_z - z
        return math.atan2(dx, dz)

    def check_obstacle(self):
        """检查前方是否有障碍物（简单避障）"""
        # so0-so7 是前方传感器
        front_values = [self.sonars[i].getValue() for i in range(8)]
        # Pioneer3dx 距离传感器值越小表示越近
        min_dist = min(front_values) if front_values else 1000
        return min_dist < 0.5  # 0.5 米内有障碍

    # -------- 状态处理 --------

    def handle_idle(self):
        """空闲状态：轮询任务文件"""
        self.stop()

        if self.step_count % POLL_INTERVAL_STEPS != 0:
            return

        task = consume_task()
        if task is None:
            return

        self.current_task = task
        pos = task["target_position"]
        # task.json 中 target_position 为 [x, y, z]，Webots 地面平面是 x-z
        self.target_x = pos[0]
        self.target_z = pos[2] if len(pos) > 2 else pos[1]

        self.state = STATE_NAVIGATING
        task_name = task.get("target", "未知目标")
        print(f"[PatrolController] 收到任务: {task_name}")
        print(f"[PatrolController] 目标位置: ({self.target_x:.2f}, {self.target_z:.2f})")

    def handle_navigating(self):
        """导航状态：向目标移动"""
        dist = self.distance_to_target()

        # 到达判定
        if dist < ARRIVE_THRESHOLD:
            self.stop()
            self.state = STATE_INSPECTING
            self.inspect_start_time = self.robot.getTime()
            x, z = self.get_position()
            print(f"[PatrolController] 到达目标! 当前位置: ({x:.2f}, {z:.2f}), 距离: {dist:.3f}m")
            print(f"[PatrolController] 开始模拟巡检...")
            return

        # 计算方向偏差
        target_bearing = self.bearing_to_target()
        current_heading = self.get_heading()
        angle_error = normalize_angle(target_bearing - current_heading)

        # 简单避障
        if self.check_obstacle():
            # 遇到障碍物，尝试右转绕行
            self.set_speed(MAX_SPEED * 0.3, -MAX_SPEED * 0.3)
            return

        # 比例控制导航
        turn = TURN_KP * angle_error
        forward = FORWARD_KP * min(dist, 1.0)

        # 角度偏差大时原地转，偏差小时边走边转
        if abs(angle_error) > 0.5:
            # 原地旋转对准目标
            self.set_speed(turn, -turn)
        else:
            # 前进 + 微调方向
            left_speed = forward + turn
            right_speed = forward - turn
            self.set_speed(left_speed, right_speed)

    def handle_inspecting(self):
        """巡检状态：模拟拍照检测"""
        elapsed = self.robot.getTime() - self.inspect_start_time
        if elapsed >= INSPECTION_DURATION:
            self.state = STATE_DONE
            print(f"[PatrolController] 模拟巡检完成 ({INSPECTION_DURATION}s)")
        # 巡检期间保持静止
        self.stop()

    def handle_done(self):
        """完成状态：写入结果并回到空闲"""
        x, z = self.get_position()
        task = self.current_task

        result = {
            "task_id": task.get("task_id", 0),
            "target": task.get("target", "unknown"),
            "status": "completed",
            "defect_type": task.get("defect_type", ""),
            "action": task.get("action", "inspect"),
            "robot_position": [round(x, 3), 0.0, round(z, 3)],
            "target_position": task.get("target_position", []),
            "inspection_result": {
                "photo_taken": True,
                "defect_detected": True if task.get("defect_type") else False,
                "defect_detail": task.get("defect_type", "无异常"),
                "confidence": 0.92,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"已完成对 {task.get('target', '目标')} 的巡检",
        }

        save_result(result)
        print(f"[PatrolController] 结果已写入: {RESULT_FILE}")

        self.current_task = None
        self.state = STATE_IDLE
        print("[PatrolController] 等待下一个任务...\n")

    # -------- 主循环 --------

    def run(self):
        """控制器主循环"""
        print("[PatrolController] 启动巡检控制器，等待任务...")

        handlers = {
            STATE_IDLE: self.handle_idle,
            STATE_NAVIGATING: self.handle_navigating,
            STATE_INSPECTING: self.handle_inspecting,
            STATE_DONE: self.handle_done,
        }

        while self.robot.step(self.timestep) != -1:
            self.step_count += 1
            handler = handlers.get(self.state, self.handle_idle)
            handler()


# ============ 入口 ============

if __name__ == "__main__":
    controller = PatrolController()
    controller.run()
