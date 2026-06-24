# 实验八：LBS 蒙皮 (Linear Blend Skinning)

> 计算机图形学实验 · 北京师范大学  
> 授课教师：张鸿文　　助教：张怡冉

---

## 实验目标

基于 **SMPL 参数化人体模型**，完整复现 LBS 蒙皮的四个阶段，并进行可视化：

| 阶段 | 内容 |
|------|------|
| (a) | 模板网格 $\bar{T}$ 与蒙皮权重 $\mathcal{W}$ |
| (b) | 形状校正后网格 $\bar{T} + B_S(\beta)$ 及关节回归 $J(\beta)$ |
| (c) | 姿态校正后网格 $T_P(\beta,\theta) = \bar{T} + B_S(\beta) + B_P(\theta)$ |
| (d) | 经过 LBS 变换后的最终姿态 |

---

## 环境配置

```bash
conda create -n cg-lbs python=3.10 -y
conda activate cg-lbs
pip install torch numpy matplotlib smplx
```

## 模型文件

将 SMPL 模型文件放置在以下路径（文件需自行从官网下载，不随代码分发）：

```
models/smpl/SMPL_NEUTRAL.pkl
```

> 下载地址：[smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de)（需注册，仅供学术使用）

---

## 文件结构

```
.
├── run_lbs_lab.py       # 基础实验：LBS 四阶段可视化 + 手写实现验证
├── animate_march.py     # 选作：原地踏步动画生成
├── animate_lbs.py       # 选作：单关节旋转动画生成
├── models/
│   └── smpl/
│       └── SMPL_NEUTRAL.pkl   # ← 需自行下载，不包含在仓库中
└── outputs/             # 运行后自动生成
    ├── stage_a_template_weights.png
    ├── stage_b_shaped_joints.png
    ├── stage_c_pose_offsets.png
    ├── stage_d_lbs_result.png
    ├── comparison_grid.png
    ├── all_joint_weights.png
    ├── summary.txt
    ├── lbs_march.gif        # 选作输出
    ├── march_preview.png    # 选作输出
    └── lbs_animation.gif    # 选作输出
```

---

## 基础实验：`run_lbs_lab.py`

### 运行

```bash
# Windows
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python run_lbs_lab.py --model-dir ./models --out-dir ./outputs --joint-id 18

# Linux / macOS
KMP_DUPLICATE_LIB_OK=TRUE python run_lbs_lab.py --model-dir ./models --out-dir ./outputs --joint-id 18
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-dir` | `./models` | 模型目录 |
| `--out-dir` | `./outputs` | 输出目录 |
| `--joint-id` | `18` | 可视化权重热力图的关节编号（0–23） |
| `--num-betas` | `10` | 使用的形状参数维度 |

### 输出说明

**`stage_a_template_weights.png`**  
模板网格（T-pose）+ 指定关节的蒙皮权重热力图。颜色越亮表示该关节对该区域影响越强。

**`stage_b_shaped_joints.png`**  
加入形状参数 $\beta$ 后的网格，以及由关节回归器从形变后网格计算得到的关节位置。

**`stage_c_pose_offsets.png`**  
加入姿态混合形变 $B_P(\theta)$ 后的网格，颜色表示各顶点姿态偏移量的大小。弯曲部位（肘、膝）颜色最明显。

**`stage_d_lbs_result.png`**  
完成 LBS 变换后的最终人体姿态，关节位置为运动学链变换后的实际位置。

**`comparison_grid.png`**  
四阶段 2×2 对比图，直观展示各阶段差异。

**`summary.txt`**  
模型基础信息 + 手写 LBS 与官方 `forward()` 的逐顶点误差对比：

```
num_vertices: 6890
num_faces: 13776
num_joints: 24
manual_vs_official_mean_abs_error: 0.0000000000
manual_vs_official_max_abs_error: 0.0000000000
```

<img width="1517" height="1559" alt="all_joint_weights" src="https://github.com/user-attachments/assets/1b717112-2d7f-4eee-b2a8-71daf97139cc" />
<img width="2417" height="2176" alt="comparison_grid" src="https://github.com/user-attachments/assets/827d0684-a4d4-46a1-b26c-2dc35ba5ac07" />
手写实现与官方结果完全一致，验证了 LBS 复现的正确性。

---

## 选作：姿态动画

### 原地踏步动画：`animate_march.py`

固定形状参数，驱动髋、膝、肩、肘关节做自然走步循环，导出 GIF 和 MP4。

```bash
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python animate_march.py --model-dir ./models --out-dir ./outputs --highlight-weights
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--frames` | `48` | 总帧数（覆盖一个完整步伐周期）|
| `--fps` | `20` | GIF / MP4 帧率 |
| `--highlight-weights` | 关闭 | 叠加髋+膝综合蒙皮权重热力图 |

**动画设计要点：**

- 髋关节绕 X 轴屈伸 ±0.30 rad，左右反相
- 膝关节用 $\frac{1-\cos(t)}{2}$ 曲线保证只屈不过伸，相位滞后髋约 0.55 rad
- 肩关节与对侧髋同相摆动（正常走路的手臂交叉规律）
- 骨盆加入轻微侧倾（±0.06 rad）消除僵硬感
- GIF 设 `loop=0`，帧序列首尾衔接，天然无缝循环

输出文件：

```
outputs/lbs_march.gif        # 无缝循环 GIF
outputs/lbs_march.mp4        # H.264 视频（需要 ffmpeg）
outputs/march_preview.png    # 8 帧静态预览，标注步态相位
```
<img width="570" height="795" alt="lbs_march" src="https://github.com/user-attachments/assets/f8c832a3-ff27-4c38-9477-8ba65eceddbd" />

### 单关节旋转动画：`animate_lbs.py`

让指定关节从 0 旋转到目标角度再归零，ping-pong 循环，便于观察单个关节的蒙皮影响范围。

```bash
python animate_lbs.py --model-dir ./models --out-dir ./outputs \
    --joint-id 18 --axis 1 --max-angle 1.8 --frames 48 --highlight-weights
```

| 参数 | 说明 |
|------|------|
| `--joint-id` | 旋转关节编号（0–23）|
| `--axis` | 旋转轴：0=X，1=Y，2=Z |
| `--max-angle` | 最大旋转角度（弧度）|

---

## SMPL 关节索引参考

| ID | 关节名 | ID | 关节名 |
|----|--------|----|--------|
| 0 | pelvis | 12 | neck |
| 1 | left_hip | 13 | left_collar |
| 2 | right_hip | 14 | right_collar |
| 4 | left_knee | 15 | head |
| 5 | right_knee | 16 | left_shoulder |
| 7 | left_ankle | 17 | right_shoulder |
| 8 | right_ankle | 18 | left_elbow |
| 10 | left_foot | 19 | right_elbow |
| 11 | right_foot | 20 | left_wrist |
<img width="1517" height="1559" alt="all_joint_weights" src="https://github.com/user-attachments/assets/48530eaf-63bf-457a-8415-808232f1a8fd" />
<img width="838" height="975" alt="lbs_animation" src="https://github.com/user-attachments/assets/7ae7862f-d249-440e-b3b5-9b9ad59ad4c7" />

---

## 思考题回答摘要

**为什么一个顶点不只受一个关节影响？**  
关节边界附近的顶点需要平滑过渡，若只绑定单个关节，关节弯曲时皮肤会出现尖锐折叠（"candy-wrapper"问题）。多关节加权混合能产生自然的皮肤拉伸效果，这也是 Linear Blend Skinning 命名的由来。

**为什么关节位置要从形变后的网格回归？**  
人体骨骼位置与体型直接相关——高挑的人肩关节位置更高，肥胖的人髋关节间距更宽。若关节固定不变，形变后的皮肤和骨骼位置会产生明显错位。

**为什么 LBS 之前还需要加姿态混合形变 $B_P(\theta)$？**  
纯刚体旋转无法表达皮肤在关节弯曲时产生的体积变化（肘部弯曲时肌肉凸出、膝部弯曲时皮肤皱褶等）。姿态混合形变通过线性组合预学习的形变基，补偿这些纯 LBS 无法表达的几何细节。<img width="838" height="975" alt="lbs_animation" src="https://github.com/user-attachments/assets/c3b85039-857c-4c08-a018-18d4d7a95265" />
