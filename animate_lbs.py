"""
animate_lbs.py  ——  LBS 蒙皮姿态动画（选作）
----------------------------------------------
固定 shape 参数，让指定关节从 0 → target_angle → 0 循环旋转，
观察蒙皮权重区域如何随骨骼运动被平滑带动。

用法：
    python animate_lbs.py --model-dir ./models --out-dir ./outputs \
        --joint-id 18 --axis 1 --max-angle 1.8 --frames 48 --fps 24 \
        --highlight-weights          # 叠加权重热力图（可选）
"""

import os, sys, types, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

import smplx
from smplx.lbs import (
    blend_shapes, vertices2joints,
    batch_rodrigues, batch_rigid_transform,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── chumpy shim（与 run_lbs_lab.py 相同） ──────────────────────────────────
class _ChumpyArrayShim:
    def __setstate__(self, state): self.__dict__.update(state)
    def _array(self):
        if hasattr(self, "r"): return self.r
        if hasattr(self, "x"): return self.x
        raise AttributeError
    def __array__(self, dtype=None): return np.asarray(self._array(), dtype=dtype)
    @property
    def shape(self): return np.asarray(self).shape
    def __len__(self): return len(np.asarray(self))
    def __getitem__(self, item): return np.asarray(self)[item]

def install_chumpy_shim():
    if "chumpy.ch" in sys.modules: return
    cm = types.ModuleType("chumpy"); ch = types.ModuleType("chumpy.ch")
    _ChumpyArrayShim.__name__ = _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    ch.Ch = _ChumpyArrayShim; cm.ch = ch
    sys.modules["chumpy"] = cm; sys.modules["chumpy.ch"] = ch

# ── 几何工具 ───────────────────────────────────────────────────────────────
def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def smpl_to_plot(pts):          # SMPL Y-up → matplotlib Z-up
    return pts[:, [0, 2, 1]]

def set_axes_equal(ax, verts):
    mn, mx = verts.min(0), verts.max(0)
    c, r = (mn+mx)/2, 0.5*np.max(mx-mn+1e-8)
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)

def face_colors_from_scalar(scalar, faces, cmap_name="plasma"):
    s = (scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-8)
    return plt.get_cmap(cmap_name)(s[faces].mean(1))

def shade(verts, faces, fc):
    tris = verts[faces]
    n = np.cross(tris[:,1]-tris[:,0], tris[:,2]-tris[:,0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    light = np.array([-0.25, -0.55, 0.80]); light /= np.linalg.norm(light)
    I = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    out = fc.copy(); out[:,:3] *= I[:,None]; return out

def draw_frame(ax, verts_np, faces, joints_np,
               weight_scalar=None, title="", elev=12, azim=108):
    pv = smpl_to_plot(verts_np)
    pj = smpl_to_plot(joints_np)

    if weight_scalar is not None:
        fc = face_colors_from_scalar(weight_scalar, faces, "plasma")
    else:
        fc = np.tile([0.82, 0.67, 0.52, 1.0], (faces.shape[0], 1))

    fc = shade(pv, faces, fc)
    mesh = Poly3DCollection(pv[faces], facecolors=fc,
                            linewidths=0.03, edgecolors=(0,0,0,0.04))
    ax.add_collection3d(mesh)
    ax.scatter(*pj.T, c="white", s=14, depthshade=False,
               edgecolors="black", linewidths=0.35, zorder=5)
    set_axes_equal(ax, pv)
    ax.set_proj_type("persp", focal_length=0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=2)

# ── LBS 单帧计算 ────────────────────────────────────────────────────────────
def prepare_posedirs(pd, expected):
    if pd.dim() != 2: pd = pd.reshape(pd.shape[0], -1)
    if pd.shape[0] == expected: return pd
    if pd.shape[1] == expected: return pd.T
    raise RuntimeError(f"posedirs shape mismatch: {tuple(pd.shape)}")

def lbs_frame(model, betas, global_orient, body_pose):
    """返回 (verts, J_transformed, pose_offset_norm)，均为 numpy。"""
    dtype, device = betas.dtype, betas.device
    vt = model.v_template.unsqueeze(0) if model.v_template.dim()==2 else model.v_template
    shapedirs = model.shapedirs[:, :, :betas.shape[1]]
    v_shaped = vt + blend_shapes(betas, shapedirs)
    J = vertices2joints(model.J_regressor, v_shaped)

    full_pose = torch.cat([global_orient, body_pose], dim=1)
    rot_mats = batch_rodrigues(full_pose.view(-1,3)).view(1,-1,3,3)
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feat = (rot_mats[:,1:,:,:] - ident).view(1,-1)
    posedirs = prepare_posedirs(model.posedirs, pose_feat.shape[1])
    v_posed = v_shaped + torch.matmul(pose_feat, posedirs).view(1,-1,3)

    J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents, dtype=dtype)
    nj = J.shape[1]
    W = model.lbs_weights.unsqueeze(0).expand(1,-1,-1)
    T = torch.matmul(W, A.view(1,nj,16)).view(1,-1,4,4)
    hom = torch.cat([v_posed, torch.ones((1,v_posed.shape[1],1),dtype=dtype,device=device)], dim=2)
    verts = torch.matmul(T, hom.unsqueeze(-1))[:,:,:3,0]

    pose_off_norm = np.linalg.norm(to_np((v_posed - v_shaped)[0]), axis=1)
    return to_np(verts[0]), to_np(J_transformed[0]), pose_off_norm

# ── 主程序 ─────────────────────────────────────────────────────────────────
def build_shape(device, dtype, num_betas=10):
    betas = torch.zeros((1,num_betas), dtype=dtype, device=device)
    vals = [2.0, -1.2, 0.8]
    for i, v in enumerate(vals[:num_betas]): betas[0,i] = v
    return betas

SMPL_JOINT_NAMES = [
    "pelvis","left_hip","right_hip","spine1",
    "left_knee","right_knee","spine2",
    "left_ankle","right_ankle","spine3",
    "left_foot","right_foot","neck",
    "left_collar","right_collar","head",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hand","right_hand",
]

def main(args):
    device = torch.device("cpu"); dtype = torch.float32
    out_dir = os.path.join(SCRIPT_DIR, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    frames_dir = os.path.join(out_dir, "anim_frames")
    os.makedirs(frames_dir, exist_ok=True)

    install_chumpy_shim()
    model_dir = os.path.join(SCRIPT_DIR, args.model_dir) if not os.path.isabs(args.model_dir) else args.model_dir
    model = smplx.create(model_path=model_dir, model_type="smpl",
                         gender="neutral", ext="pkl",
                         num_betas=args.num_betas).to(device)
    faces = np.asarray(model.faces, dtype=np.int32)
    betas = build_shape(device, dtype, args.num_betas)

    joint_id   = args.joint_id
    axis_idx   = args.axis       # 0=x, 1=y, 2=z
    max_angle  = args.max_angle
    n_frames   = args.frames
    highlight  = args.highlight_weights

    jname = SMPL_JOINT_NAMES[joint_id] if joint_id < len(SMPL_JOINT_NAMES) else f"joint_{joint_id}"
    axis_name = ["X","Y","Z"][axis_idx]
    weight_scalar = to_np(model.lbs_weights[:, joint_id]) if highlight else None

    # 角度序列：0 → max → 0（ping-pong，导出 gif 天然循环）
    half = n_frames // 2
    angles = np.concatenate([
        np.linspace(0, max_angle, half, endpoint=False),
        np.linspace(max_angle, 0, n_frames - half),
    ])

    frame_paths = []
    print(f"生成 {n_frames} 帧动画：关节 {joint_id}({jname})，旋转轴 {axis_name}，最大角度 {max_angle:.2f} rad")

    for i, angle in enumerate(angles):
        global_orient = torch.zeros((1,3), dtype=dtype, device=device)
        body_pose = torch.zeros((1,23*3), dtype=dtype, device=device)

        # 仅旋转目标关节（global=0 不算在 body_pose 里，body_pose 索引从 joint1 开始）
        if joint_id >= 1:
            start = (joint_id - 1) * 3
            body_pose[0, start + axis_idx] = angle
        else:
            global_orient[0, axis_idx] = angle

        with torch.no_grad():
            verts_np, joints_np, pose_off_norm = lbs_frame(model, betas, global_orient, body_pose)

        fig = plt.figure(figsize=(7, 7), facecolor="#0d0d0d")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")

        deg = np.degrees(angle)
        title_str = (
            f"Joint {joint_id} ({jname})  |  Axis {axis_name}  |  {deg:+.1f}°\n"
            f"Frame {i+1}/{n_frames}"
        )
        draw_frame(ax, verts_np, faces, joints_np,
                   weight_scalar=weight_scalar,
                   title=title_str, elev=10, azim=105)
        ax.title.set_color("#e8e8e8")

        # 底部信息条
        fig.text(0.5, 0.02,
                 f"LBS Animation  ·  Fixed shape β=[2.0,−1.2,0.8,…]  ·  "
                 f"{'Weight heatmap ON' if highlight else 'Default shading'}",
                 ha="center", va="bottom", fontsize=7, color="#666666")

        path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        frame_paths.append(path)

        if (i+1) % 8 == 0 or i == 0:
            print(f"  [{i+1:3d}/{n_frames}]  angle={deg:+.1f}°")

    # ── 导出 GIF ──────────────────────────────────────────────────────────
    gif_path = os.path.join(out_dir, "lbs_animation.gif")
    with imageio.get_writer(gif_path, mode="I",
                            duration=1.0/args.fps, loop=0) as writer:
        for p in frame_paths:
            writer.append_data(imageio.imread(p))
    print(f"\n✓ GIF 已保存：{gif_path}")

    # ── 尝试导出 MP4（需要 ffmpeg） ────────────────────────────────────────
    try:
        import subprocess, shutil
        if shutil.which("ffmpeg"):
            mp4_path = os.path.join(out_dir, "lbs_animation.mp4")
            cmd = [
                "ffmpeg", "-y", "-framerate", str(args.fps),
                "-i", os.path.join(frames_dir, "frame_%04d.png"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", mp4_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ MP4 已保存：{mp4_path}")
        else:
            print("（未检测到 ffmpeg，跳过 MP4 导出）")
    except Exception as e:
        print(f"（MP4 导出失败：{e}）")

    # ── 生成一张静态多帧预览图 ─────────────────────────────────────────────
    preview_indices = np.linspace(0, n_frames-1, 8, dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9),
                             facecolor="#0d0d0d",
                             subplot_kw={"projection":"3d"})
    fig.subplots_adjust(hspace=0.05, wspace=0.0)

    for ax, idx in zip(axes.flat, preview_indices):
        ax.set_facecolor("#0d0d0d")
        global_orient = torch.zeros((1,3), dtype=dtype, device=device)
        body_pose = torch.zeros((1,23*3), dtype=dtype, device=device)
        ang = angles[idx]
        if joint_id >= 1:
            body_pose[0, (joint_id-1)*3 + axis_idx] = ang
        else:
            global_orient[0, axis_idx] = ang
        with torch.no_grad():
            verts_np, joints_np, _ = lbs_frame(model, betas, global_orient, body_pose)
        draw_frame(ax, verts_np, faces, joints_np,
                   weight_scalar=weight_scalar,
                   title=f"{np.degrees(ang):+.0f}°", elev=10, azim=105)
        ax.title.set_color("#cccccc")

    fig.suptitle(
        f"LBS Skinning Animation Preview  ·  Joint {joint_id} ({jname})  ·  Axis {axis_name}",
        color="#e0e0e0", fontsize=13, y=0.97
    )
    preview_path = os.path.join(out_dir, "anim_preview.png")
    fig.savefig(preview_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ 预览图已保存：{preview_path}")
    print("\n完成！输出文件：")
    print(f"  {gif_path}")
    print(f"  {preview_path}")
    print(f"  {frames_dir}/*.png  ({n_frames} 帧)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LBS 姿态动画生成器")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--out-dir",   default="./outputs")
    parser.add_argument("--joint-id",  type=int,   default=18,
                        help="要旋转的关节编号（SMPL 0-23）")
    parser.add_argument("--axis",      type=int,   default=1,
                        choices=[0,1,2],
                        help="旋转轴：0=X  1=Y  2=Z")
    parser.add_argument("--max-angle", type=float, default=1.8,
                        help="最大旋转角度（弧度）")
    parser.add_argument("--frames",    type=int,   default=48,
                        help="总帧数（ping-pong，GIF 自动循环）")
    parser.add_argument("--fps",       type=int,   default=20,
                        help="GIF / MP4 帧率")
    parser.add_argument("--num-betas", type=int,   default=10)
    parser.add_argument("--highlight-weights", action="store_true",
                        help="用 plasma 热力图叠加目标关节的蒙皮权重")
    args = parser.parse_args()
    main(args)