# 机器人操作评测

TidyBot 色标任务适合检查模型接入、截图和动作闭环。对机械臂操作能力的评估使用独立的公开任务、动作接口和仿真判分器。本项目的第一组外部对照使用 Robo Use 0.6.1，无需修改上游场景。

## 公开结果对应的实验

- Kimi 与 LiberAI 的 2026 年 11 月 24 日案例中，Kimi K3.1 Pro 负责科研闭环。LiberAI 用既有动作模型、训练和评测管线验证迭代，报告单项仿真任务一周内从 45% 到 95%。公开案例摘要未列出能重建这组数字的任务定义、初始权重、分割与种子。LiberAI 的 Liber-0 技术文章另行描述了用带动作监督的人类数据、机器人数据和本体对齐来训练世界动作模型。 [Kimi 客户案例](https://www.kimi.com/customers/) · [Liber-0 技术文章](https://www.liberai.com/blog/liber0/)
- Anthropic 的 Fable 5.2 发布文报告 RoboDojo 双臂仿真内部对照的平均成功率为 64.9%，使用 Max 推理档位。公开的 RoboProbe L3 参考接口提供 RGB、任务指令和机器人自身状态，模型发出世界坐标系下的末端目标，非学习控制器负责运动。这个公开接口可复现一类实验协议，Anthropic 的数字对应其内部同 Harness 比较。 [Anthropic 发布文](https://www.anthropic.com/claude-fable-and-mythos-5-2) · [RoboProbe L3 接口](https://github.com/RoboProbe/RoboProbe/blob/main/docs/minimal_harness.md)
- GPT-6.2 发布文的 CAD、3D 资产和由模型编写的物理求解器展示了设计与工具使用能力。公开 RoboDojo L3 接口则给出了同系列模型作为闭环机器人控制策略的另一条证据。机器人评分依赖完成任务的物理结果。 [GPT-6.2 发布文](https://openai.com/index/introducing-gpt-6-2-astra-sol/) · [RoboDojo 协议说明](https://github.com/benchflow-ai/robouse/blob/main/docs/suites/robodojo.md)
- PEAR 展示了一条有公开代码的模型训练路线。在 LIBERO 和 RoboDojo 上用仿真强化学习训练开放权重 27B 推理模型；模型直接生成末端位姿目标。复现实验需要分别记录公开 checkpoint 复测和从基础模型重训。 [论文](https://arxiv.org/abs/2611.18562) · [代码](https://github.com/mit-han-lab/pear)

## 本地协议

使用 Robo Use 的两个原样视觉任务。`arc-gravity-vision` 要求从样例推断方块移动规则并用机械臂执行，`arc-stack-by-size-vision` 要求按大小堆叠真实仿真方块。官方 oracle 可以使用特权状态，作为任务可解性对照。noop 检查起始场景是否会被误判为成功。视觉模型经 Show-Harness 提交动作，得分来自上游仿真器。任务、模型、Harness、时间预算和输出目录均记录在每次运行的配置中。

### 复现命令

使用独立的 Python 3.12 环境，Linux 无窗口时使用 EGL。安装 Robo Use 后运行其自带诊断。

```bash
python3.12 -m venv .venv-robouse
source .venv-robouse/bin/activate
python -m pip install 'robouse[sim]==0.6.1'
export MUJOCO_GL=egl
robouse doctor

run_dir=$(mktemp -d)
for task in arc-gravity-vision arc-stack-by-size-vision; do
    robouse run --task "$task" --harness oracle --out "$run_dir/oracle"
    robouse run --task "$task" --harness noop --out "$run_dir/noop"
done

printf '{"mode":"zs"}\n' >"$run_dir/qwen-zs.json"
ROBOUSE_SHOW_HARNESS_CONFIG="$run_dir/qwen-zs.json" \
ROBOUSE_SHOW_HARNESS_URL=http://127.0.0.1:11434/v1 \
robouse run --task arc-gravity-vision --harness show-harness \
  --model qwen3-vl:2b-instruct --timeout 150 --out "$run_dir/qwen"
```

查看 stdout 的分数以及运行目录内的 `verifier/reward.txt`、`episode/trace.jsonl`、相机图和录像。区分零动作、提出动作未执行、接触失败和完成错误布局。单次任务的结果适合排查接口和动作决策，多模型比较需要固定同一组布局、相机权限、动作空间和预算。

## 本地参考结果

| 上游任务 | Oracle 分数 / 仿真步数 | Noop 分数 / 仿真步数 |
| --- | --- | --- |
| arc-gravity-vision | 1.0 / 999 | 0.0 / 0 |
| arc-stack-by-size-vision | 1.0 / 465 | 0.0 / 0 |

Qwen3-VL 2B Instruct 在 `arc-gravity-vision` 的零样本试验中发出了 `MV_FWD`，执行 8 个物理控制步，在 150 秒的模型时限内未完成，得分 0.0。此前使用 Show-Harness 自托管默认模式的试跑在 0 步时发出 `DONE`，也得 0.0。零样本结果同时受模型与时间预算限制，无法用它估计无时间约束下的成功率。

这两个任务使用 Robo Use 的 MuJoCo 环境。RoboDojo 官方仿真使用 Isaac Sim 和双臂 ARX X5。主机经实测是 RTX 4060 Laptop、8188 MiB 显存，低于 Isaac Sim 5.1 官方所列的 16 GB 最低规格。RoboDojo 原版分数需要在对应硬件和环境中复跑。 [Isaac Sim 5.1 硬件要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

## 配对消融

首轮问题是模型在墙钟预算内能否执行足够多的决策。先前零样本试验到时仅完成 8 个控制步，因此冻结一个单变量实验。比较 Show-Harness 零样本默认配置与只关闭高层 `subgoal` 插件的配置。后者保留相机、动作空间、执行控制器和其余插件。假设是少做一层子目标推理能在相同时间内推进更多有效动作，主指标仍是上游任务完成分。推进步数、部分进度和结束原因用于诊断。

用 `arc-gravity-vision` 作开发任务，用 `arc-stack-by-size-vision` 作留出任务。留出任务的官方参考策略此前用于确认评测可解，模型输出未用于设定配置。每组使用相同模型、任务种子 0 和 90 秒墙钟预算。留出组反转运行顺序，减少固定顺序带来的预热影响。小样本结果用于排查，成功率比较需要更多布局与重复。

在项目根目录运行，先激活前述 Robo Use 虚拟环境并确保 Ollama 已载入模型。

```bash
pair_dir=$(mktemp -d)
python benchmarks/robouse_pair.py --out "$pair_dir/run"
```

脚本先写 `protocol.json` 和两份配置，然后顺序执行四个试验。每轮完成即更新 `summary.json`，同时保留 Robo Use 原始结果、模型决策日志、执行轨迹和录像。可以通过 `--model` 切换本地 4B，通过 `--seed` 或 `--timeout` 开启新一组独立实验。对比时应保持两组使用同一模型配置和预算。

| 任务 | 配置 | 得分 | 模型决策 / 控制步 | 空抓 | 位移不足 1 mm |
| --- | --- | ---: | ---: | ---: | ---: |
| 推断方块移动规则 | 默认 | 0.0 | 8 / 296 | 7 | 0 |
| 推断方块移动规则 | 关闭子目标 | 0.0 | 7 / 287 | 7 | 0 |
| 按大小堆叠 | 默认 | 0.0 | 19 / 134 | 1 | 6 |
| 按大小堆叠 | 关闭子目标 | 0.0 | 40 / 134 | 1 | 27 |

开发组两次均超时。留出组默认配置超时，关闭子目标的配置达到单阶段决策上限后放弃。四轮均启用了图像输入。单纯关闭子目标未提高得分，留出任务增加的决策大部分消耗在原地或边界动作上。这一组轨迹支持单独测试动作结果反馈，例如空抓后要求重新定位，位移不足时切换观察或动作。该改动应与本轮消融分开比较。

## 视觉分层执行验收

`benchmarks/vision_stack.py` 使用同一个上游 `arc-stack-by-size-vision` 任务、末端控制接口、800 步上限和最终物理判分。Qwen 从公开样例和相机输入判断尺寸规则。视觉模块从标定顶视 RGB 图分割三种方块颜色，估计位置与尺寸。控制器将模型推断的规则转换成有界抓取和堆叠动作，检查闭爪后的持物反馈。上游评测在释放并稳定后判定结果。策略不读取仿真物体位置或官方参考解。

本地 Qwen3-VL 2B Instruct 对种子 0、1、2 均通过，每次使用 442 个控制步。此上游任务的默认布局固定，三个重复试验验证了重置后的可重复性，覆盖范围是该布局和已知的三个方块颜色。上一节的自由动作对照仍保留，不能将两种 Harness 的结果用作仅比较模型能力的分数。

```bash
stack_dir=$(mktemp -d)
python benchmarks/vision_stack.py --seed 0 --seed 1 --seed 2 --out "$stack_dir/run"
```

每次运行保存相机图、推断规则、测得的几何数据、仿真轨迹、录像和上游结果。任意一次未通过时命令返回非零退出码。
