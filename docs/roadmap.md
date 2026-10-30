# 仿真与机器人学习路线

当前代码只是高层 VLM 生成动作计划的 turtlesim 原型；以下是研究路线，尚未实现或验证。

1. **Gazebo + ROS 2**：先将相机、里程计和速度控制桥接到统一接口，记录图像、位姿、动作与时间戳；增加简单障碍物场景。Humble 推荐 Gazebo Fortress；Harmonic 的组合需要额外处理软件包冲突，见 [Gazebo 兼容表](https://gazebosim.org/docs/latest/ros_installation/)。先建立固定种子的任务集，测量成功率、碰撞率、超时率和推理延迟。
2. **视觉闭环**：采集仿真相机图像，使用检测/分割及深度估计结合相机标定和 TF，将观测转换到机器人坐标系。VLM 给语义目标，规划与碰撞检测交给 Nav2/MoveIt；执行一小步后重新观测。不能仅凭 VLM 猜测距离或坐标。
3. **MuJoCo + IL/RL**：实现与 Gazebo 相同的 observation/action 适配器，录制专家轨迹；以行为克隆建立基线，再使用 PPO/SAC 等策略做仿真训练与对比。增加物体、光照、质量随机化，并分开训练和评估场景。
4. **Transformer / VLA**：基于轨迹学习图像、语言、状态到动作的策略，和当前规则动作库比较；RT-2 是可参考的 [VLA 示例](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/)。当前 Ollama VLM 生成离散计划不等同于训练好的 VLA 策略。
5. **世界模型 / WAM**：研究用动作条件预测未来观测或状态，再进行模型预测控制；评估预测误差和长时滚动误差，不把生成视频当作精确物理仿真。可参考 [Genie](https://deepmind.google/research/publications/60474/)。
6. **Isaac Lab 3.0**：先确认实际 GPU、显存、驱动和 RAM，再决定 Newton 无 Isaac Sim 或完整 Isaac Sim 路线。不同版本要求不同；[3.0 文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html)列出完整 Isaac Sim 工作流的 Python 3.12、32 GB RAM 和 16 GB 显存要求。消费级 4060 机器需要先核实具体显存，不预设能运行完整栈。

跨仿真器共享动作语义与数据格式，但各环境的动力学、传感器、碰撞和坐标定义仍需分别校准。实物部署前还需要独立的硬件急停、速度/碰撞保护、权限隔离和系统级验证。
