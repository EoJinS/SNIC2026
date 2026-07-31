
나는 sionna library를 활용해서 mobility가 있는 유저(UE)의 움직임에 따라 Base station과의 dominant path 가 달라지는 것을 visualize 하고 싶어. 아래 Task에 따라 예시 코드(.ipynb)를 작성해줘.

sionna에서 scene load (scene은 상관 없음. sionna에서 다운받아 쓸 수 있는 scene 아무거나 쓰기.)
BS 1개, UE 1개 설정
BS, UE 사이의 path들 시각화 (dominant path 는 더 굵은 빨간색 선으로 잘 보이게 하기)
UE가 움직임에 따라 BS와의 path , 특히 dominant path가 변하는 것을 시각화하기.
위의 task 중 네가 생각하기에 수정이 필요한 부분이 있다면 수정해서 코드를 짜도 돼.

sionna mobility 관련 document : https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html

간략히 보기

재시도를 위해 누락된 매개변수를 식별했습니다
재시도를 위해 누락된 매개변수를 식별했습니다
https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html
nvlabs.github.io


sionna rt Paths object vertices attribute API reference
결과 9개


Extract interaction points and corresponding objects in the ray tracer · NVlabs/sionna · Discussion #329
github.com

Paths — Sionna 2.0.1
nvlabs.github.io

Sionna RT: Technical Report
arxiv.org
hermespy.channel.sionna_rt_channel - HermesPy
hermespy.org

sionna.rt.scene — Sionna 2.0.1
nvlabs.github.io

Getting output information from each ray in sionna · NVlabs/sionna · Discussion #505
github.com

batch number for paths.types · NVlabs/sionna · Discussion #533
github.com

Introduction to Sionna RT — Sionna 2.0.1
nvlabs.github.io

API Documentation — Sionna 1.2.1 documentation
nvlabs.github.io

prep workspace
프로그래매틱 방식으로 노트북 생성하여 포맷 오류 제거하기로 결정했다.


명령 실행 중

Install nbformat and build the notebook via script
재시도를 위해 누락된 매개변수를 식별했습니다.


Create python script that builds the ipynb notebook via nbformat
bash

cat << 'PYEOF' > /home/claude/build_nb.py
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Sionna RT: UE Mobility에 따른 Dominant Path 변화 시각화

이 노트북은 Sionna의 Ray Tracing(RT) 모듈을 이용해 다음을 수행합니다.

1. Sionna 내장 예제 씬(scene)을 로드
2. Base Station(BS) 1개, User Equipment(UE) 1개 배치
3. BS-UE 사이의 propagation path들을 시각화 (dominant path는 굵은 빨간색 선으로 강조)
4. UE가 이동함에 따라 path, 특히 dominant path가 어떻게 바뀌는지 애니메이션으로 시각화

> 참고: Sionna RT의 `scene.render()`는 모든 path를 한 번에 그려주지만, path별로 색/굵기를
> 개별적으로 지정하는 기능은 제공하지 않습니다. 따라서 이 노트북에서는
> `Paths` 객체에서 `vertices`(상호작용 지점 좌표)와 `a`(경로 이득), `interactions`(상호작용 종류)를
> 직접 추출하여 matplotlib으로 커스텀 시각화를 만듭니다. 이를 통해 dominant path를
> 원하는 대로 굵은 빨간 선으로 강조할 수 있습니다.
> Sionna 자체 렌더링(`scene.render`)도 참고용으로 함께 보여줍니다.

Reference: https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html
"""
))

cells.append(nbf.v4.new_code_cell(
"""# ------------------------------------------------------------
# 0. 환경 설정 & 임포트
# ------------------------------------------------------------
%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import sionna.rt
except ImportError:
    os.system("pip install sionna-rt")
    import sionna.rt

from sionna.rt import (
    load_scene, PlanarArray, Transmitter, Receiver, Camera,
    PathSolver,
)
from sionna.rt.constants import InteractionType

no_preview = True  # True: scene.render()로 정적 이미지 생성 (headless 환경에서 안전)

print("Sionna version:", sionna.rt.__version__ if hasattr(sionna.rt, "__version__") else "unknown")
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 1. Scene 로드

Sionna RT가 기본 제공하는 예제 씬 중 `simple_street_canyon_with_cars`를 사용합니다.
건물과 주차된 차량들이 있어 반사(reflection)/회절(diffraction)이 발생하기 좋은 환경이라
UE가 이동할 때 dominant path가 바뀌는 모습을 관찰하기에 적합합니다.
(어떤 씬을 쓰든 상관없다고 하셨으므로, 필요하면 `sionna.rt.scene.<다른 씬 이름>`으로 교체하면 됩니다.)
"""
))

cells.append(nbf.v4.new_code_cell(
"""# ------------------------------------------------------------
# 1. Scene 로드
# ------------------------------------------------------------
scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars, merge_shapes=False)

# 안테나 배열 설정 (BS/UE 모두 단일 등방성 안테나 -> path 해석을 단순화)
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

# 가능하면 synthetic array를 사용하도록 설정 (Paths 텐서 shape을 단순하게 유지)
try:
    scene.synthetic_array = True
except Exception:
    pass

cam = Camera(position=[0, -80, 90], look_at=[0, 0, 0])

if no_preview:
    scene.render(camera=cam, resolution=[700, 500]);
else:
    scene.preview()
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2. BS 1개, UE 1개 배치

- **BS**: 건물 옆, 지상에서 약 15m 높이에 고정 (거리의 한쪽 끝)
- **UE**: 도로를 따라 이동하는 보행자/차량 단말, 초기 위치는 BS 반대쪽 끝 부근
"""
))

cells.append(nbf.v4.new_code_cell(
"""# ------------------------------------------------------------
# 2. BS(Transmitter), UE(Receiver) 배치
# ------------------------------------------------------------
bs_pos = [-32.0, 8.0, 15.0]     # BS: 건물 벽면 근처, 높은 곳에 설치된 안테나
ue_start_pos = [25.0, -5.0, 1.5]  # UE 초기 위치: 도로 위, 보행자/단말 높이

scene.add(Transmitter("bs", position=bs_pos))
scene.add(Receiver("ue", position=ue_start_pos))

print("BS position:", scene.get('bs').position.numpy()[:, 0])
print("UE position:", scene.get('ue').position.numpy()[:, 0])

if no_preview:
    scene.render(camera=cam, resolution=[700, 500]);
else:
    scene.preview()
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 3. Path 계산 및 시각화 (Dominant path 강조)

`PathSolver`로 BS-UE 사이의 propagation path를 계산합니다. 이후 아래 두 가지 방식으로 시각화합니다.

1. **Sionna 내장 렌더링** (`scene.render(paths=paths)`) — 3D 씬 위에 모든 path를 표시 (참고용)
2. **커스텀 matplotlib 3D 플롯** — `Paths.a`(경로 이득)로부터 dominant path를 찾아
   굵은 빨간 선으로, 나머지 path는 얇은 회색 선으로 표시

Dominant path는 "경로 이득의 크기(|a|) 제곱, 즉 수신 전력이 가장 큰 path"로 정의합니다.
"""
))

cells.append(nbf.v4.new_code_cell(
"""# ------------------------------------------------------------
# 3-1. Path 계산
# ------------------------------------------------------------
p_solver = PathSolver()

paths = p_solver(
    scene=scene,
    max_depth=3,        # LoS + 최대 3회 반사/회절까지 고려
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=False,
    diffraction=True,   # 코너 회절도 고려 (LoS가 막혔을 때 대체 경로)
)

# Sionna 내장 렌더링 (참고용) - 모든 경로가 함께 표시됨
if no_preview:
    scene.render(camera=cam, paths=paths, resolution=[700, 500]);
else:
    scene.preview(paths=paths)
"""
))

cells.append(nbf.v4.new_code_cell(
"""# ------------------------------------------------------------
# 3-2. Paths 객체에서 dominant path 및 path geometry 추출
# ------------------------------------------------------------

def extract_path_data(paths):
    \"\"\"Paths 객체에서 이후 시각화에 필요한 정보를 numpy array로 뽑아낸다.

    반환값:
        vertices      : (max_depth, num_paths, 3)  각 path의 상호작용 지점 좌표
        interactions  : (max_depth, num_paths)      상호작용 종류 (0 = 없음/LoS)
        power_db      : (num_paths,)                path별 수신 전력 (dB, 상대값)
        valid         : (num_paths,) bool           유효한(존재하는) path 여부
        dominant_idx  : int                         dominant path의 index
    \"\"\"
    verts = np.asarray(paths.vertices)          # (depth, rx, tx, num_paths, 3) 혹은 유사 shape
    inter = np.asarray(paths.interactions)      # (depth, rx, tx, num_paths)
    a = np.asarray(paths.a)                     # (rx, [rx_ant,] tx, [tx_ant,] num_paths[, time])
    if not np.iscomplexobj(a):
        # 일부 버전은 (real, imag)를 튜플/스택으로 반환 -> 마지막 축이 2인 경우 처리
        if a.shape[-1] == 2:
            a = a[..., 0] + 1j * a[..., 1]

    # rx/tx(=크기 1) 축 제거
    verts = np.squeeze(verts, axis=tuple(i for i in range(1, verts.ndim - 2) if verts.shape[i] == 1))
    inter = np.squeeze(inter, axis=tuple(i for i in range(1, inter.ndim - 1) if inter.shape[i] == 1))

    num_paths = inter.shape[-1]

    # a 는 rx/tx/antenna 조합이 여러 개 섞여 있을 수 있으므로 path 축 기준으로 power를 합산
    a_flat = a.reshape(-1, num_paths)
    power = np.sum(np.abs(a_flat) ** 2, axis=0)  # (num_paths,)

    valid = power > 0
    if not np.any(valid):
        raise RuntimeError("유효한 path가 없습니다 (BS-UE 사이에 도달 가능한 path가 없음).")

    power_db = 10 * np.log10(power + 1e-30)
    dominant_idx = int(np.argmax(power))

    return verts, inter, power_db, valid, dominant_idx


def path_polyline(tx_pos, rx_pos, verts, inter, p):
    \"\"\"path index p에 대한 [TX -> 상호작용 지점들 -> RX] 좌표 리스트 반환\"\"\"
    pts = [np.asarray(tx_pos, dtype=float)]
    max_depth = verts.shape[0]
    for d in range(max_depth):
        if inter[d, p] != InteractionType.NONE:
            pts.append(verts[d, p])
    pts.append(np.asarray(rx_pos, dtype=float))
    return np.stack(pts, axis=0)


def path_type_str(inter, p):
    \"\"\"path의 상호작용 타입을 사람이 읽기 쉬운 문자열로 변환\"\"\"
    depth = int(np.sum(inter[:, p] != InteractionType.NONE))
    if depth == 0:
        return "LoS"
    types = []
    for d in range(inter.shape[0]):
        t = int(inter[d, p])
        if t == InteractionType.NONE:
            continue
        elif t == InteractionType.SPECULAR:
            types.append("반사")
        elif t == InteractionType.DIFFUSE:
            types.append("확산반사")
        elif t == InteractionType.REFRACTION:
            types.append("굴절")
        elif t == InteractionType.DIFFRACTION:
            types.append("회절")
        else:
            types.append(f"type{t}")
    return f"{depth}차 " + "+".join(types)


verts, inter, power_db, valid, dominant_idx = extract_path_data(paths)
num_paths = inter.shape[-1]

print(f"전체 path 후보 수: {num_paths}, 유효 path 수: {int(np.sum(valid))}")
for p in range(num_paths):
    if not valid[p]:
        continue
    marker = " <-- DOMINANT" if p == dominant_idx else ""
    print(f"  path {p:2d}: {path_type_str(inter, p):12s}  power = {power_db[p]:6.1f} dB{marker}")
"""
))

cells.append(nbf.v4.new_code_cell(
"""# ------------------------------------------------------------
# 3-3. 커스텀 3D 시각화 (dominant path = 굵은 빨간 선)
# ------------------------------------------------------------

def plot_paths_3d(ax, bs_pos, ue_pos, verts, inter, valid, dominant_idx, title=""):
    ax.clear()
    num_paths = inter.shape[-1]

    for p in range(num_paths):
        if not valid[p]:
            continue
        pts = path_polyline(bs_pos, ue_pos, verts, inter, p)
        if p == dominant_idx:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color="red", linewidth=4.0, zorder=10,
                    label="Dominant path")
            ax.scatter(pts[1:-1, 0], pts[1:-1, 1], pts[1:-1, 2],
                       color="darkred", s=40, zorder=11)
        else:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color="steelblue", linewidth=1.2, alpha=0.5, zorder=2)

    ax.scatter(*bs_pos, color="black", marker="^", s=180, zorder=12, label="BS")
    ax.scatter(*ue_pos, color="green", marker="o", s=120, zorder=12, label="UE")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")


fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
plot_paths_3d(
    ax, bs_pos, ue_start_pos, verts, inter, valid, dominant_idx,
    title=f"BS-UE Paths (dominant: {path_type_str(inter, dominant_idx)}, "
          f"{power_db[dominant_idx]:.1f} dB)"
)
plt.tight_layout()
plt.show()
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 4. UE 이동에 따른 Dominant Path 변화 시각화

UE를 도로를 따라 일직선으로 이동시키면서(step마다 위치를 바꾼 뒤 path를 재계산),
각 스텝에서의 path 구조와 dominant path를 기록하고 애니메이션으로 만듭니다.

- 각 스텝마다 `scene.get("ue").position`을 갱신하고 `PathSolver`를 다시 호출합니다.
  (Mobility 튜토리얼에서 소개된 두 가지 방식 중, "물체를 실제로 이동시키고 path를
  다시 트레이싱하는" 정석적인 방법을 사용합니다.)
- LoS가 건물/차량에 막히는 구간에서는 반사/회절 경로가 dominant path가 되는 것을
  관찰할 수 있습니다.
"""
))

cells.append(nbf.v4.new_code_cell(
"""# ------------------------------------------------------------
# 4-1. UE 이동 경로 정의 & 스텝별 path 계산
# ------------------------------------------------------------
num_steps = 20
ue_traj_x = np.linspace(25.0, -25.0, num_steps)   # BS 반대편 -> BS 근처로 이동
ue_traj = np.stack([ue_traj_x,
                     np.full(num_steps, ue_start_pos[1]),
                     np.full(num_steps, ue_start_pos[2])], axis=1)

frames = []  # 각 스텝의 (ue_pos, verts, inter, power_db, valid, dominant_idx)

for i, pos in enumerate(ue_traj):
    scene.get("ue").position = pos.tolist()
    paths_i = p_solver(
        scene=scene, max_depth=3, los=True,
        specular_reflection=True, diffuse_reflection=False,
        refraction=False, diffraction=True,
    )
    verts_i, inter_i, power_db_i, valid_i, dom_i = extract_path_data(paths_i)
    frames.append(dict(
        ue_pos=pos.copy(), verts=verts_i, inter=inter_i,
        power_db=power_db_i, valid=valid_i, dominant_idx=dom_i,
    ))
    print(f"[step {i:2d}] UE x={pos[0]:6.1f}  dominant = {path_type_str(inter_i, dom_i):10s} "
          f"({power_db_i[dom_i]:6.1f} dB)")
"""
))

cells.append(nbf.v4.new_code_cell(
"""# ------------------------------------------------------------
# 4-2. 스텝별 스냅샷을 격자로 한눈에 비교
# ------------------------------------------------------------
show_steps = list(range(0, num_steps, max(1, num_steps // 6)))[:6]

fig = plt.figure(figsize=(18, 11))
for k, i in enumerate(show_steps):
    fr = frames[i]
    ax = fig.add_subplot(2, 3, k + 1, projection="3d")
    plot_paths_3d(
        ax, bs_pos, fr["ue_pos"], fr["verts"], fr["inter"], fr["valid"], fr["dominant_idx"],
        title=f"step {i}: dominant={path_type_str(fr['inter'], fr['dominant_idx'])} "
              f"({fr['power_db'][fr['dominant_idx']]:.0f} dB)"
    )
plt.tight_layout()
plt.show()
"""
))

cells.append(nbf.v4.new_code_cell(
"""# ------------------------------------------------------------
# 4-3. 애니메이션 (GIF) 생성: UE 이동에 따른 dominant path 변화
# ------------------------------------------------------------
from matplotlib.animation import FuncAnimation, PillowWriter

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

def update(i):
    fr = frames[i]
    plot_paths_3d(
        ax, bs_pos, fr["ue_pos"], fr["verts"], fr["inter"], fr["valid"], fr["dominant_idx"],
        title=f"step {i}: dominant = {path_type_str(fr['inter'], fr['dominant_idx'])}  "
              f"({fr['power_db'][fr['dominant_idx']]:.1f} dB)"
    )
    return ax,

ani = FuncAnimation(fig, update, frames=num_steps, interval=500, blit=False)

output_path = "/mnt/user-data/outputs/ue_mobility_dominant_path.gif"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
ani.save(output_path, writer=PillowWriter(fps=2))
plt.close(fig)

print("Saved animation to:", output_path)
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 요약

- `PathSolver`로 계산된 `Paths` 객체에서 `vertices`(상호작용 지점), `interactions`(상호작용 종류),
  `a`(복소 경로 이득)를 추출하여 path별 수신 전력을 계산했습니다.
- 수신 전력이 가장 큰 path를 **dominant path**로 정의하고, matplotlib 3D 플롯에서
  굵은 빨간 선으로 강조했습니다.
- UE를 도로를 따라 이동시키며 매 스텝마다 path를 재계산(re-tracing)한 결과,
  LoS가 건물/차량에 막히는 구간에서는 반사/회절 경로로 dominant path가 전환되는 것을
  확인할 수 있습니다.
- 더 짧은 시간/거리 스케일(수 파장 이내)의 이동에 대해서는, 매 스텝 path를 다시 트레이싱하는
  대신 `Paths.doppler`를 이용한 Doppler-shift 기반 시간 evolution(Mobility 튜토리얼의 두 번째 방법)을
  사용하면 훨씬 빠르게 채널의 시간 변화를 근사할 수 있습니다.
"""
))

nb['cells'] = cells

with open('/home/claude/ue_mobility_dominant_path.ipynb', 'w') as f:
    nbf.write(nb, f)

print("notebook written")
PYEOF
pip install nbformat --break-system-packages -q
python3 /home/claude/build_nb.py
