# LLM-Robot

用本地 Ollama 模型将自然语言和可选图片转换成经过校验的动作序列，在 ROS 2 turtlesim 中执行。原课程项目说明保存在 [docs/README-2024.md](docs/README-2024.md)。

当前是仿真原型：动作只有 `move`、`rotate`、`stop`；支持顺序执行、速度和时长限制、位姿超时停止。模型不会直接生成代码，也不会把图中位置作为可执行坐标。模型可能误解意图，请先检查输出；这个控制器不适用于实物机器人。

## 环境

Ubuntu 22.04 对应 ROS 2 Humble。可使用[鱼香 ROS](https://github.com/fishros/install) 安装，按界面选择 ROS 2 Humble 桌面版；安装过程会请求 sudo 密码，注意阅读换源选项。另需 `python3-colcon-common-extensions` 和 `ros-humble-turtlesim`：

```bash
sudo apt install python3-colcon-common-extensions ros-humble-turtlesim
source /opt/ros/humble/setup.bash
ros2 run turtlesim turtlesim_node
```

安装 [Ollama](https://docs.ollama.com/linux)，启动服务后获取模型：

```bash
ollama pull qwen3-vl:4b
```

如果 Ollama 安装在用户目录并配置了用户级 systemd 服务，可以用 `systemctl --user status ollama` 查看状态，使用 `systemctl --user start ollama` 启动。该服务随用户会话启动。

默认 `qwen3-vl:4b`，可设置一行 `export OLLAMA_MODEL=qwen3-vl:2b` 切换；选用其他模型时要求该模型支持 Ollama 的结构化输出，使用 `--image` 时还需支持视觉。`OLLAMA_HOST` 可配置服务地址。模型需要分别下载。

使用系统 Python 3.10 构建（请退出可能影响 ROS 的 conda 环境）：

```bash
cd /path/to/LLM-Robot
source /opt/ros/humble/setup.bash
colcon build --packages-select rosgpt
source install/setup.bash
```

## 运行

在已 source 两个 setup 文件的终端启动 turtlesim 和控制器：

```bash
ros2 launch rosgpt demo.launch.py
```

另开一个已 source 的终端先预览，再按需执行。第二次会重新请求模型，`--execute` 会直接发布当次生成的结果；两次结果可能不同。

```bash
ros2 run rosgpt rosgpt '前进一米，逆时针转90度'
ros2 run rosgpt rosgpt '前进一米，逆时针转90度' --execute
```

视觉输入示例（需要有视觉能力的模型）：

```bash
ros2 run rosgpt rosgpt '如果图片里箭头指向左边，逆时针旋转45度；否则停止。' --image examples/left_arrow.png
```

可将图片路径替换为自己的图片，并添加 `--execute` 执行生成的动作。图片经 Ollama 提供给模型作为上下文，不是相机实时订阅；没有实现目标定位、抓取或导航。模型可能忽略约束或识别错误。控制器忙时拒绝新计划，紧急停止可绕过模型：

```bash
ros2 topic pub --once /voice_cmd std_msgs/msg/String '{data: "{\"actions\":[{\"action\":\"stop\"}]}"}'
```

停止依赖 ROS 通信与控制器进程，只用于此仿真原型。位姿停止更新、动作超时或收到 stop 会发布零速度并丢弃剩余队列；进程异常退出、网络通信中断无法保证可靠停车。系统不应直连真实机器人。

在不安装 ROS 的情况下执行单元测试：

```bash
python3 -m unittest discover -s tests -v
```

真实端到端验收需要已安装 ROS 2、已构建本项目，以及正在运行的 Ollama 和 `qwen3-vl:4b`。下面的测试使用独立 ROS 域和无窗口的 turtlesim，不会操作已有的演示窗口：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
/usr/bin/python3 tests/ros_acceptance.py
```

验收会实际测试中文命令、图片命令、位姿变化和运行中停止。模型输出具有不确定性，运行时间也取决于机器和首次加载时间。

未来可扩展方向及约束见 [docs/roadmap.md](docs/roadmap.md)。
