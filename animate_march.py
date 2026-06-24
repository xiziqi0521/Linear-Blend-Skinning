"""
animate_march.py  ——  LBS 蒙皮：原地踏步动画
---------------------------------------------
固定 shape 参数，驱动髋/膝/肩/肘做自然走步循环，
导出 GIF + MP4。

用法：
    python animate_march.py --model-dir ./models --out-dir ./outputs
    python animate_march.py --model-dir ./models --out-dir ./outputs --highlight-weights
"""

import os, sys, types, argparse, subprocess, shutil
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

# ── chumpy shim ──────────────────────────────────────────────────────────────
class _ChumpyArrayShim:
    def __setstate__(self, s): self.__dict__.update(s)
    def _array(self):
        if hasattr(self,"r"): return self.r
        if hasattr(self,"x"): return self.x
        raise AttributeError
    def __array__(self,dtype=None): return np.asarray(self._array(),dtype=dtype)
    @property
    def shape(self): return np.asarray(self).shape
    def __len__(self): return len(np.asarray(self))
    def __getitem__(self,i): return np.asarray(self)[i]

def install_chumpy_shim():
    if "chumpy.ch" in sys.modules: return
    cm=types.ModuleType("chumpy"); ch=types.ModuleType("chumpy.ch")
    _ChumpyArrayShim.__name__=_ChumpyArrayShim.__qualname__="Ch"
    _ChumpyArrayShim.__module__="chumpy.ch"
    ch.Ch=_ChumpyArrayShim; cm.ch=ch
    sys.modules["chumpy"]=cm; sys.modules["chumpy.ch"]=ch

# ── 几何工具 ─────────────────────────────────────────────────────────────────
def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def smpl_to_plot(pts):
    return pts[:,[0,2,1]]          # SMPL Y-up → matplotlib Z-up

def set_axes_equal(ax, v):
    mn,mx=v.min(0),v.max(0)
    c,r=(mn+mx)/2, 0.5*np.max(mx-mn+1e-8)
    ax.set_xlim(c[0]-r,c[0]+r)
    ax.set_ylim(c[1]-r,c[1]+r)
    ax.set_zlim(c[2]-r,c[2]+r)

def face_colors_from_scalar(scalar, faces, cmap="plasma"):
    s=(scalar-scalar.min())/(scalar.max()-scalar.min()+1e-8)
    return plt.get_cmap(cmap)(s[faces].mean(1))

def shade(verts,faces,fc):
    t=verts[faces]
    n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-8
    L=np.array([-0.3,-0.5,0.8]); L/=np.linalg.norm(L)
    I=0.35+0.65*np.clip(n@L,0,1)
    out=fc.copy(); out[:,:3]*=I[:,None]; return out

def draw_frame(ax, verts_np, faces, joints_np,
               weight_scalar=None, title="", elev=10, azim=108):
    pv=smpl_to_plot(verts_np); pj=smpl_to_plot(joints_np)
    if weight_scalar is not None:
        fc=face_colors_from_scalar(weight_scalar,faces)
    else:
        fc=np.tile([0.80,0.64,0.48,1.0],(faces.shape[0],1))
    fc=shade(pv,faces,fc)
    ax.add_collection3d(Poly3DCollection(
        pv[faces], facecolors=fc, linewidths=0.02, edgecolors=(0,0,0,0.04)))
    ax.scatter(*pj.T, c="white", s=12, depthshade=False,
               edgecolors="black", linewidths=0.3, zorder=5)
    set_axes_equal(ax,pv)
    ax.set_proj_type("persp",focal_length=0.9)
    ax.view_init(elev=elev,azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=2)

# ── LBS ──────────────────────────────────────────────────────────────────────
def prepare_posedirs(pd, expected):
    if pd.dim()!=2: pd=pd.reshape(pd.shape[0],-1)
    if pd.shape[0]==expected: return pd
    if pd.shape[1]==expected: return pd.T
    raise RuntimeError(f"posedirs shape {tuple(pd.shape)} vs expected {expected}")

def lbs_frame(model, betas, global_orient, body_pose):
    dtype,device=betas.dtype,betas.device
    vt=model.v_template
    if vt.dim()==2: vt=vt.unsqueeze(0)
    shapedirs=model.shapedirs[:,:,:betas.shape[1]]
    v_shaped=vt+blend_shapes(betas,shapedirs)
    J=vertices2joints(model.J_regressor,v_shaped)
    full_pose=torch.cat([global_orient,body_pose],dim=1)
    rot_mats=batch_rodrigues(full_pose.view(-1,3)).view(1,-1,3,3)
    ident=torch.eye(3,dtype=dtype,device=device)
    pf=(rot_mats[:,1:,:,:]-ident).view(1,-1)
    pd=prepare_posedirs(model.posedirs,pf.shape[1])
    v_posed=v_shaped+torch.matmul(pf,pd).view(1,-1,3)
    J_tr,A=batch_rigid_transform(rot_mats,J,model.parents,dtype=dtype)
    nj=J.shape[1]
    W=model.lbs_weights.unsqueeze(0).expand(1,-1,-1)
    T=torch.matmul(W,A.view(1,nj,16)).view(1,-1,4,4)
    hom=torch.cat([v_posed,torch.ones((1,v_posed.shape[1],1),dtype=dtype,device=device)],2)
    verts=torch.matmul(T,hom.unsqueeze(-1))[:,:,:3,0]
    return to_np(verts[0]), to_np(J_tr[0])

# ── 自然踏步姿态生成器 ────────────────────────────────────────────────────────
def make_march_pose(t: float, style: str = "natural"):
    """
    t ∈ [0, 2π) 对应一个完整步伐周期（左右各一步）。

    SMPL body_pose layout（各关节占 3 个轴角分量，X/Y/Z）：
        joint_id  body_pose 起始 idx
        1 left_hip        0
        2 right_hip       3
        4 left_knee       9
        5 right_knee     12
        16 left_shoulder  45
        17 right_shoulder 48
        18 left_elbow     51
        19 right_elbow    54
    """
    pose = np.zeros(23 * 3)

    # ── 幅度参数 ──────────────────────────────────────────
    HIP_FLEX   = 0.30    # 髋屈伸 ±rad
    KNEE_FLEX  = 0.38    # 膝屈曲 peak rad（只屈，不过伸）
    KNEE_PHASE = 0.55    # 膝相位滞后（rad）
    SHLD_SWING = 0.18    # 肩摆动 ±rad（与对侧髋同相）
    ELBOW_MID  = 0.40    # 肘静息屈曲
    ELBOW_AMP  = 0.15    # 肘随肩额外摆动

    # ── 加入轻微骨盆倾斜，让踏步更自然 ──────────────────
    PELVIS_TILT = 0.06   # 骨盆左右倾斜 ±rad（绕Z）
    pose[2] =  PELVIS_TILT * np.sin(t)   # pelvis Z （left_hip parent）

    # ── 髋关节：绕 X 轴屈伸，左右反相 ────────────────────
    left_hip_x  =  HIP_FLEX * np.sin(t)
    right_hip_x = -HIP_FLEX * np.sin(t)
    pose[0] = left_hip_x           # left_hip  X
    pose[3] = right_hip_x          # right_hip X

    # 髋轻微内收/外展（绕 Z），幅度很小，增加自然感
    HIP_ABD = 0.04
    pose[2 + 0] +=  HIP_ABD * np.cos(t)   # left_hip  Z（叠加）
    pose[2 + 3] += -HIP_ABD * np.cos(t)   # right_hip Z

    # ── 膝关节：只屈不伸，相位滞后 ───────────────────────
    # knee_flex = peak * 0.5*(1 - cos(t - phase))，始终 ≥ 0
    left_knee_x  = KNEE_FLEX * 0.5 * (1 - np.cos(t          - KNEE_PHASE))
    right_knee_x = KNEE_FLEX * 0.5 * (1 - np.cos(t + np.pi  - KNEE_PHASE))
    pose[9]  = left_knee_x
    pose[12] = right_knee_x

    # ── 肩关节：与对侧髋同相（正常走路手臂摆动） ─────────
    left_shld_x  = -SHLD_SWING * np.sin(t)   # 与右髋同相
    right_shld_x =  SHLD_SWING * np.sin(t)   # 与左髋同相
    pose[45] = left_shld_x
    pose[48] = right_shld_x

    # ── 肘关节：静息屈曲 + 随肩小幅变化 ──────────────────
    pose[51] = ELBOW_MID + ELBOW_AMP * np.sin(t)
    pose[54] = ELBOW_MID - ELBOW_AMP * np.sin(t)

    return pose

# ── 主程序 ────────────────────────────────────────────────────────────────────
def build_shape(device, dtype, num_betas=10):
    b=torch.zeros((1,num_betas),dtype=dtype,device=device)
    for i,v in enumerate([2.0,-1.2,0.8][:num_betas]): b[0,i]=v
    return b

def main(args):
    device=torch.device("cpu"); dtype=torch.float32
    out_dir=(os.path.join(SCRIPT_DIR,args.out_dir)
             if not os.path.isabs(args.out_dir) else args.out_dir)
    frames_dir=os.path.join(out_dir,"march_frames")
    os.makedirs(frames_dir,exist_ok=True)

    install_chumpy_shim()
    model_dir=(os.path.join(SCRIPT_DIR,args.model_dir)
               if not os.path.isabs(args.model_dir) else args.model_dir)
    model=smplx.create(model_path=model_dir,model_type="smpl",
                       gender="neutral",ext="pkl",
                       num_betas=args.num_betas).to(device)
    faces=np.asarray(model.faces,dtype=np.int32)
    betas=build_shape(device,dtype,args.num_betas)

    n_frames=args.frames
    fps=args.fps
    highlight=args.highlight_weights

    # 权重热力图：同时显示髋+膝的综合影响
    if highlight:
        highlight_joints=[1,2,4,5]   # left/right hip + knee
        w=to_np(model.lbs_weights)
        weight_scalar=w[:,highlight_joints].sum(axis=1)
        weight_scalar=(weight_scalar-weight_scalar.min())/(weight_scalar.max()-weight_scalar.min()+1e-8)
    else:
        weight_scalar=None

    # 相位序列，覆盖两个完整步伐周期（GIF loop=0 无缝衔接）
    phases=np.linspace(0,2*np.pi,n_frames,endpoint=False)

    print(f"生成 {n_frames} 帧原地踏步动画 (fps={fps})…")
    frame_paths=[]

    for i,t in enumerate(phases):
        pose_np=make_march_pose(t)
        global_orient=torch.zeros((1,3),dtype=dtype,device=device)
        body_pose=torch.tensor(pose_np,dtype=dtype,device=device).unsqueeze(0)

        with torch.no_grad():
            verts_np,joints_np=lbs_frame(model,betas,global_orient,body_pose)

        fig=plt.figure(figsize=(5,7),facecolor="#111111")
        ax=fig.add_subplot(111,projection="3d",facecolor="#111111")
        draw_frame(ax,verts_np,faces,joints_np,
                   weight_scalar=weight_scalar,
                   title=f"Marching in Place  |  Frame {i+1}/{n_frames}",
                   elev=8, azim=110)
        ax.title.set_color("#dddddd")
        fig.text(0.5,0.01,
                 "LBS Skinning  ·  Fixed β=[2.0,−1.2,0.8,…]  ·  "
                 "Hip/Knee/Shoulder/Elbow",
                 ha="center",fontsize=6.5,color="#555555")

        p=os.path.join(frames_dir,f"frame_{i:04d}.png")
        fig.savefig(p,dpi=140,bbox_inches="tight",facecolor=fig.get_facecolor())
        plt.close(fig)
        frame_paths.append(p)
        if (i+1)%8==0 or i==0:
            print(f"  [{i+1:3d}/{n_frames}]  phase={np.degrees(t):.1f}°")

    # ── GIF ──────────────────────────────────────────────────────────────────
    gif_path=os.path.join(out_dir,"lbs_march.gif")
    with imageio.get_writer(gif_path,mode="I",duration=1.0/fps,loop=0) as w:
        for p in frame_paths: w.append_data(imageio.imread(p))
    print(f"\n✓ GIF  →  {gif_path}")

    # ── MP4 ──────────────────────────────────────────────────────────────────
    if shutil.which("ffmpeg"):
        mp4_path=os.path.join(out_dir,"lbs_march.mp4")
        cmd=["ffmpeg","-y","-framerate",str(fps),
             "-i",os.path.join(frames_dir,"frame_%04d.png"),
             "-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v","libx264","-pix_fmt","yuv420p","-crf","18",mp4_path]
        subprocess.run(cmd,check=True,capture_output=True)
        print(f"✓ MP4  →  {mp4_path}")
    else:
        print("（未检测到 ffmpeg，跳过 MP4）")

    # ── 静态预览 8 帧 ─────────────────────────────────────────────────────────
    idxs=np.linspace(0,n_frames-1,8,dtype=int)
    fig,axes=plt.subplots(2,4,figsize=(20,9),facecolor="#111111",
                          subplot_kw={"projection":"3d"})
    fig.subplots_adjust(hspace=0.05,wspace=0.0)
    step_labels=["L-strike","L-mid","R-lift","R-swing",
                 "R-strike","R-mid","L-lift","L-swing"]
    for ax,idx,lbl in zip(axes.flat,idxs,step_labels):
        ax.set_facecolor("#111111")
        t=phases[idx]
        pose_np=make_march_pose(t)
        global_orient=torch.zeros((1,3),dtype=dtype,device=device)
        body_pose=torch.tensor(pose_np,dtype=dtype,device=device).unsqueeze(0)
        with torch.no_grad():
            verts_np,joints_np=lbs_frame(model,betas,global_orient,body_pose)
        draw_frame(ax,verts_np,faces,joints_np,
                   weight_scalar=weight_scalar,
                   title=lbl, elev=8, azim=110)
        ax.title.set_color("#bbbbbb")
    fig.suptitle("LBS Marching in Place — 8-Frame Preview",
                 color="#e0e0e0",fontsize=13,y=0.97)
    prev_path=os.path.join(out_dir,"march_preview.png")
    fig.savefig(prev_path,dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ 预览图  →  {prev_path}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="LBS 原地踏步动画")
    parser.add_argument("--model-dir",default="./models")
    parser.add_argument("--out-dir",  default="./outputs")
    parser.add_argument("--frames",   type=int,  default=48,
                        help="总帧数（建议 48 或 64）")
    parser.add_argument("--fps",      type=int,  default=20)
    parser.add_argument("--num-betas",type=int,  default=10)
    parser.add_argument("--highlight-weights",action="store_true",
                        help="用 plasma 热力图叠加髋+膝综合蒙皮权重")
    args=parser.parse_args()
    main(args)