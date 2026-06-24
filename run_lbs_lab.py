import os
import argparse
import sys
import types
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import smplx
from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class _ChumpyArrayShim:
    """Minimal pickle shim for old SMPL files that stored arrays as chumpy.Ch."""

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _array(self):
        if hasattr(self, "r"):
            return self.r
        if hasattr(self, "x"):
            return self.x
        raise AttributeError("Cannot recover array data from chumpy pickle object")

    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)

    @property
    def shape(self):
        return np.asarray(self).shape

    def __len__(self):
        return len(np.asarray(self))

    def __getitem__(self, item):
        return np.asarray(self)[item]

def install_chumpy_pickle_shim():
    """Allow pickle.load to read legacy SMPL .pkl files without installing chumpy."""
    if "chumpy.ch" in sys.modules:
        return

    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")

    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    chumpy_ch_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module

    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module

def make_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def resolve_script_path(path: str):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def set_axes_equal(ax, vertices: np.ndarray):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * np.max(maxs - mins + 1e-8)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

def get_face_colors_from_vertex_scalar(vertex_scalar: np.ndarray, faces: np.ndarray, cmap_name="viridis"):
    scalar = vertex_scalar.astype(np.float64)
    scalar = (scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-8)
    face_scalar = scalar[faces].mean(axis=1)
    cmap = plt.get_cmap(cmap_name)
    return cmap(face_scalar)

def get_face_colors_from_joint_weights(lbs_weights: np.ndarray, faces: np.ndarray):
    face_weights = lbs_weights[faces].mean(axis=1)
    dominant_joint = np.argmax(face_weights, axis=1)
    dominant_weight = np.max(face_weights, axis=1)

    num_joints = lbs_weights.shape[1]
    palette = plt.get_cmap("hsv")(np.linspace(0.0, 1.0, num_joints, endpoint=False))
    face_colors = palette[dominant_joint]
    strength = 0.35 + 0.65 * dominant_weight
    face_colors[:, :3] *= strength[:, None]
    face_colors[:, :3] += (1.0 - strength[:, None]) * 0.88
    face_colors[:, 3] = 1.0
    return face_colors

def smpl_to_plot_coords(points: np.ndarray):
    return points[:, [0, 2, 1]]

def shade_face_colors(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8

    light_dir = np.array([-0.25, -0.55, 0.80], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    intensity = 0.35 + 0.65 * np.clip(normals @ light_dir, 0.0, 1.0)

    shaded = face_colors.copy()
    shaded[:, :3] *= intensity[:, None]
    return shaded

def draw_mesh(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray = None,
    vertex_scalar: np.ndarray = None,
    face_colors: np.ndarray = None,
    title: str = "",
    elev: float = 12,
    azim: float = 108,
):
    plot_vertices = smpl_to_plot_coords(vertices)
    plot_joints = None if joints is None else smpl_to_plot_coords(joints)

    if face_colors is not None:
        face_colors = face_colors.copy()
    elif vertex_scalar is None:
        face_colors = np.tile(np.array([[0.82, 0.67, 0.52, 1.0]]), (faces.shape[0], 1))
    else:
        face_colors = get_face_colors_from_vertex_scalar(vertex_scalar, faces)
    face_colors = shade_face_colors(plot_vertices, faces, face_colors)

    mesh = Poly3DCollection(
        plot_vertices[faces],
        facecolors=face_colors,
        linewidths=0.03,
        edgecolors=(0.0, 0.0, 0.0, 0.05),
    )
    ax.add_collection3d(mesh)

    if joints is not None:
        ax.scatter(
            plot_joints[:, 0], plot_joints[:, 1], plot_joints[:, 2],
            c="white", s=12, depthshade=False,
            edgecolors="black", linewidths=0.3
        )

    set_axes_equal(ax, plot_vertices)
    ax.set_proj_type("persp", focal_length=0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)

def save_single_figure(path, vertices, faces, joints=None, vertex_scalar=None, title=""):
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(ax, vertices, faces, joints=joints, vertex_scalar=vertex_scalar, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def save_comparison_grid(path, data_dict, faces):
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(221, projection="3d")
    draw_mesh(
        ax1,
        data_dict["v_template"],
        faces,
        joints=data_dict["J_template"],
        vertex_scalar=data_dict["weight_scalar"],
        title="(a) Template + LBS Weights"
    )

    ax2 = fig.add_subplot(222, projection="3d")
    draw_mesh(
        ax2,
        data_dict["v_shaped"],
        faces,
        joints=data_dict["J_shaped"],
        title="(b) Shape Blend + Joint Regression"
    )

    ax3 = fig.add_subplot(223, projection="3d")
    draw_mesh(
        ax3,
        data_dict["v_posed"],
        faces,
        joints=data_dict["J_shaped"],
        vertex_scalar=data_dict["pose_offset_norm"],
        title="(c) Pose Blend Shapes"
    )

    ax4 = fig.add_subplot(224, projection="3d")
    draw_mesh(
        ax4,
        data_dict["verts"],
        faces,
        joints=data_dict["J_transformed"],
        title="(d) Final LBS Result"
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def save_all_joint_weights_figure(path, vertices, faces, joints, lbs_weights):
    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(
        ax,
        vertices,
        faces,
        joints=joints,
        face_colors=get_face_colors_from_joint_weights(lbs_weights, faces),
        title="All Joint LBS Weights",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def build_demo_shape(device, dtype, num_betas=10):
    betas = torch.zeros((1, num_betas), dtype=dtype, device=device)
    # 设置几个非零 beta，让体型变化明显一些
    if num_betas >= 1:
        betas[0, 0] = 2.0
    if num_betas >= 2:
        betas[0, 1] = -1.2
    if num_betas >= 3:
        betas[0, 2] = 0.8
    return betas

def build_demo_pose(device, dtype):
    # SMPL: global_orient = 3, body_pose = 23 * 3
    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
    body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)

    joint_names = {
        "left_hip": 1,
        "right_hip": 2,
        "left_knee": 4,
        "right_knee": 5,
        "left_shoulder": 16,
        "right_shoulder": 17,
        "left_elbow": 18,
        "right_elbow": 19,
    }

    def set_joint_pose(name, axis_angle):
        start = (joint_names[name] - 1) * 3
        body_pose[0, start:start + 3] = torch.tensor(axis_angle, dtype=dtype, device=device)

    set_joint_pose("left_shoulder", [0.0, 0.0, 0.45])
    set_joint_pose("right_shoulder", [0.0, 0.0, -0.45])
    set_joint_pose("left_elbow", [0.0, -0.35, 0.0])
    set_joint_pose("right_elbow", [0.0, 0.35, 0.0])
    set_joint_pose("left_hip", [0.25, 0.0, 0.08])
    set_joint_pose("right_hip", [-0.18, 0.0, -0.08])
    set_joint_pose("left_knee", [0.35, 0.0, 0.0])
    set_joint_pose("right_knee", [0.20, 0.0, 0.0])

    return global_orient, body_pose

def prepare_posedirs(posedirs: torch.Tensor, expected_pose_dim: int):
    """
    不同版本/实现里 posedirs 的形状有时写作 [P, V*3]，有时会以转置形式出现。
    官方 lbs() 需要的是 [P, V*3]。
    """
    if posedirs.dim() != 2:
        posedirs = posedirs.reshape(posedirs.shape[0], -1)

    if posedirs.shape[0] == expected_pose_dim:
        return posedirs
    if posedirs.shape[1] == expected_pose_dim:
        return posedirs.T

    raise RuntimeError(
        f"posedirs 形状与 pose_feature 不匹配，posedirs.shape={tuple(posedirs.shape)}, "
        f"expected_pose_dim={expected_pose_dim}"
    )

def compute_manual_lbs(model, betas, global_orient, body_pose):
    device = betas.device
    dtype = betas.dtype

    v_template = model.v_template
    if v_template.dim() == 2:
        v_template = v_template.unsqueeze(0)  # [1, V, 3]

    shapedirs = model.shapedirs[:, :, :betas.shape[1]]
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # (b) 由形状后的网格回归关节
    J = vertices2joints(model.J_regressor, v_shaped)

    # (c) 姿态校正
    full_pose = torch.cat([global_orient, body_pose], dim=1)  # [1, 24*3]
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)

    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)

    posedirs = prepare_posedirs(model.posedirs, expected_pose_dim=pose_feature.shape[1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(1, -1, 3)
    v_posed = v_shaped + pose_offsets

    # (d) 刚体层级变换 + LBS
    J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents, dtype=dtype)

    num_joints = J.shape[1]
    W = model.lbs_weights.unsqueeze(0).expand(1, -1, -1)  # [1, V, J]

    T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)

    homogen_coord = torch.ones((1, v_posed.shape[1], 1), dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)  # [1, V, 4]
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))       # [1, V, 4, 1]
    verts = v_homo[:, :, :3, 0]

    # 模板姿态下的关节，方便可视化 (a)
    J_template = vertices2joints(model.J_regressor, v_template)

    return {
        "v_template": v_template,
        "J_template": J_template,
        "v_shaped": v_shaped,
        "J_shaped": J,
        "pose_offsets": pose_offsets,
        "v_posed": v_posed,
        "J_transformed": J_transformed,
        "verts": verts,
    }

def compare_with_official_forward(model, betas, global_orient, body_pose, manual_verts):
    with torch.no_grad():
        output = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            return_verts=True,
        )
    official_verts = output.vertices
    diff = torch.abs(manual_verts - official_verts)
    mean_err = diff.mean().item()
    max_err = diff.max().item()
    return mean_err, max_err

def main(args):
    device = torch.device("cpu")
    dtype = torch.float32
    model_dir = resolve_script_path(args.model_dir)
    out_dir = resolve_script_path(args.out_dir)
    make_out_dir(out_dir)

    # 加载 SMPL
    install_chumpy_pickle_shim()
    model = smplx.create(
        model_path=model_dir,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        num_betas=args.num_betas,
    ).to(device)

    faces = np.asarray(model.faces, dtype=np.int32)
    num_vertices = model.v_template.shape[0]
    num_faces = faces.shape[0]
    num_joints = model.lbs_weights.shape[1]

    # 构造示例参数
    betas = build_demo_shape(device, dtype, num_betas=args.num_betas)
    global_orient, body_pose = build_demo_pose(device, dtype)

    # 手动复现 LBS 各阶段
    data = compute_manual_lbs(model, betas, global_orient, body_pose)

    # 与官方 forward 做数值对比
    mean_err, max_err = compare_with_official_forward(
        model, betas, global_orient, body_pose, data["verts"]
    )

    # 可视化所需数据
    joint_id = int(args.joint_id)
    if joint_id < 0 or joint_id >= model.lbs_weights.shape[1]:
        raise ValueError(
            f"joint_id 越界：{joint_id}，可选范围应为 [0, {model.lbs_weights.shape[1] - 1}]"
        )

    weight_scalar = to_numpy(model.lbs_weights[:, joint_id])
    pose_offset_norm = np.linalg.norm(to_numpy(data["pose_offsets"][0]), axis=1)

    # 保存单张图
    save_single_figure(
        os.path.join(out_dir, "stage_a_template_weights.png"),
        to_numpy(data["v_template"][0]),
        faces,
        joints=to_numpy(data["J_template"][0]),
        vertex_scalar=weight_scalar,
        title=f"(a) Template Mesh + Weight of Joint {joint_id}",
    )

    save_single_figure(
        os.path.join(out_dir, "stage_b_shaped_joints.png"),
        to_numpy(data["v_shaped"][0]),
        faces,
        joints=to_numpy(data["J_shaped"][0]),
        vertex_scalar=None,
        title="(b) Shape Blend + Joint Regression",
    )

    save_single_figure(
        os.path.join(out_dir, "stage_c_pose_offsets.png"),
        to_numpy(data["v_posed"][0]),
        faces,
        joints=to_numpy(data["J_shaped"][0]),
        vertex_scalar=pose_offset_norm,
        title="(c) Pose Blend Shapes (colored by |pose_offsets|)",
    )

    save_single_figure(
        os.path.join(out_dir, "stage_d_lbs_result.png"),
        to_numpy(data["verts"][0]),
        faces,
        joints=to_numpy(data["J_transformed"][0]),
        vertex_scalar=None,
        title="(d) Final LBS Result",
    )

    # 保存总对比图
    grid_dict = {
        "v_template": to_numpy(data["v_template"][0]),
        "J_template": to_numpy(data["J_template"][0]),
        "v_shaped": to_numpy(data["v_shaped"][0]),
        "J_shaped": to_numpy(data["J_shaped"][0]),
        "v_posed": to_numpy(data["v_posed"][0]),
        "verts": to_numpy(data["verts"][0]),
        "J_transformed": to_numpy(data["J_transformed"][0]),
        "weight_scalar": weight_scalar,
        "pose_offset_norm": pose_offset_norm,
    }
    save_comparison_grid(
        os.path.join(out_dir, "comparison_grid.png"),
        grid_dict,
        faces,
    )

    save_all_joint_weights_figure(
        os.path.join(out_dir, "all_joint_weights.png"),
        to_numpy(data["v_template"][0]),
        faces,
        to_numpy(data["J_template"][0]),
        to_numpy(model.lbs_weights),
    )

    # 保存摘要信息
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("===== SMPL LBS Lab Summary =====\n")
        f.write(f"num_vertices: {num_vertices}\n")
        f.write(f"num_faces: {num_faces}\n")
        f.write(f"num_joints(from lbs_weights): {num_joints}\n")
        f.write(f"num_betas: {args.num_betas}\n")
        f.write(f"visualized_joint_id: {joint_id}\n")
        f.write(f"manual_vs_official_mean_abs_error: {mean_err:.10f}\n")
        f.write(f"manual_vs_official_max_abs_error: {max_err:.10f}\n")

    print("运行完成。")
    print(f"顶点数: {num_vertices}")
    print(f"面片数: {num_faces}")
    print(f"关节数: {num_joints}")
    print(f"手写 LBS 与官方 forward 的平均绝对误差: {mean_err:.10f}")
    print(f"手写 LBS 与官方 forward 的最大绝对误差: {max_err:.10f}")
    print(f"结果已保存到: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./models", help="模型目录，内部应包含 smpl/SMPL_NEUTRAL.pkl")
    parser.add_argument("--out-dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--joint-id", type=int, default=18, help="要可视化权重的关节编号")
    parser.add_argument("--num-betas", type=int, default=10, help="使用多少个 shape 参数")
    args = parser.parse_args()
    main(args)
