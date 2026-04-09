import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R_scipy
from scipy.io import loadmat
from utils import read_canonical_model, load_pc, visualize_icp_result

def as_scalar(x) -> float: return float(np.asarray(x).reshape(-1)[0])
def wrap_pi(a): return (a + np.pi) % (2 * np.pi) - np.pi
def se2_mat(x, y, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]], dtype=float)

def se2_from_mat(T): return float(T[0, 2]), float(T[1, 2]), float(math.atan2(T[1, 0], T[0, 0]))
def se2_compose(a, b): return se2_from_mat(se2_mat(*a) @ se2_mat(*b))
def se2_inv(p):
    x, y, th = p
    c, s = np.cos(th), np.sin(th)
    tinv = -np.array([[c, -s], [s, c]]).T @ np.array([x, y])
    return (float(tinv[0]), float(tinv[1]), float(-th))

def se2_relative(a, b): return se2_compose(se2_inv(a), b)
def transform_points(pose, pts2):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    return (np.array([[c, -s], [s, c]]) @ pts2.T).T + np.array([x, y])

def shift_forward(pose, dx):
    x, y, th = pose
    return (x + dx * np.cos(th), y + dx * np.sin(th), th)

def plot_traj(traj, title):
    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1])
    plt.axis("equal")
    plt.grid(True)
    plt.title(title)
    plt.xlabel("x (m)"); plt.ylabel("y (m)")

# Constants
TICK_M = 0.0022
REAR_TO_CENTER_X = 0.46355 / 2.0
LIDAR_X_FROM_REAR = 0.30183
LIDAR_X_FROM_CENTER = LIDAR_X_FROM_REAR - REAR_TO_CENTER_X
LIDAR_Y_FROM_CENTER = 0.0

def load_all_npz(data_dir, seq):
    enc = np.load(os.path.join(data_dir, f"Encoders{seq}.npz"), allow_pickle=True)
    imu = np.load(os.path.join(data_dir, f"Imu{seq}.npz"), allow_pickle=True)
    hok = dict(np.load(os.path.join(data_dir, f"Hokuyo{seq}.npz"), allow_pickle=True))
    
    ranges = hok["ranges"]
    if ranges.shape[0] != 1081 and ranges.shape[1] == 1081:
        hok["ranges"] = ranges.T

    kin_path = os.path.join(data_dir, f"Kinect{seq}.npz")
    kin = np.load(kin_path, allow_pickle=True) if os.path.exists(kin_path) else None
    return enc, imu, hok, kin

def part1_odometry(enc, imu):
    counts = np.asarray(enc["counts"])
    t_enc = np.asarray(enc["time_stamps"]).reshape(-1)
    w_imu = np.asarray(imu["angular_velocity"])
    t_imu = np.asarray(imu["time_stamps"]).reshape(-1)

    if counts.shape[0] != 4 and counts.shape[1] == 4: counts = counts.T
    yaw_enc = np.interp(t_enc, t_imu, w_imu[2, :] if w_imu.shape[0] == 3 else w_imu[:, 2])

    x, y, th = 0.0, 0.0, 0.0
    poses_rear = np.zeros((t_enc.shape[0], 3), float)
    
    for i in range(1, t_enc.shape[0]):
        dt = float(t_enc[i] - t_enc[i-1])
        if dt <= 0:
            poses_rear[i] = poses_rear[i-1]
            continue

        FR, FL, RR, RL = counts[:, i]
        v = (((FR + RR) / 2.0) * TICK_M + ((FL + RL) / 2.0) * TICK_M) / (2.0 * dt)
        w = float(yaw_enc[i])

        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th = wrap_pi(th + w * dt)
        poses_rear[i] = [x, y, th]

    poses_center = np.array([shift_forward(tuple(p), REAR_TO_CENTER_X) for p in poses_rear], float)
    return t_enc, poses_rear, poses_center

def nearest_pose_by_time(t_query, t_ref, poses_ref):
    return tuple(poses_ref[int(np.argmin(np.abs(t_ref - t_query)))])

def lidar_angles(hok):
    return as_scalar(hok["angle_min"]) + as_scalar(hok["angle_increment"]) * np.arange(hok["ranges"].shape[0])

def scan_to_points(hok, scan_id, rmin=0.2, rmax=20.0, stride=1):
    ranges = hok["ranges"][:, scan_id].astype(float)
    ang = lidar_angles(hok)

    rmin = max(rmin, as_scalar(hok.get("range_min", rmin)))
    rmax = min(rmax, as_scalar(hok.get("range_max", rmax)))

    mask = np.isfinite(ranges) & (ranges > rmin) & (ranges < rmax)
    r, a = ranges[mask], ang[mask]
    pts = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    
    return pts[::stride] if stride > 1 and pts.shape[0] > 0 else pts

def perp(v): return np.stack([-v[:, 1], v[:, 0]], axis=1)

def icp_point_to_line_2d(src, dst, init=(0,0,0), max_iter=20, trim=0.7, max_corr=0.5, tol=1e-4):
    pose = init
    tree = KDTree(dst)
    prev_mse = None

    for _ in range(max_iter):
        cur = transform_points(pose, src)
        d, idx = tree.query(cur)
        good = d < max_corr
        if np.count_nonzero(good) < 40: break

        cur_g, src_g, idx_g, d_g = cur[good], src[good], idx[good], d[good]

        # Trimming
        k = max(40, int(trim * cur_g.shape[0]))
        sel = np.argsort(d_g)[:k]
        cur_g, src_g, idx_g, d_g = cur_g[sel], src_g[sel], idx_g[sel], d_g[sel]

        q = dst[idx_g]
        q1 = dst[np.clip(idx_g - 1, 0, dst.shape[0]-1)]
        q2 = dst[np.clip(idx_g + 1, 0, dst.shape[0]-1)]

        # Filter depth jumps
        valid = (np.linalg.norm(q - q1, axis=1) < 0.3) & (np.linalg.norm(q - q2, axis=1) < 0.3)
        q, q1, q2, cur_g = q[valid], q1[valid], q2[valid], cur_g[valid]

        v_line = q2 - q1
        v_line /= np.linalg.norm(v_line, axis=1, keepdims=True) + 1e-12
        n = np.stack([-v_line[:,1], v_line[:,0]], axis=1)

        r = np.sum(n * (cur_g - q), axis=1)
        jp = perp(cur_g)
        J = np.column_stack([np.sum(n * jp, axis=1), n[:,0], n[:,1]])

        # Regularize with odom prior
        J_aug = np.vstack((J, np.eye(3) * 15.0))
        r_aug = np.concatenate((-r, np.zeros(3)))

        delta, *_ = np.linalg.lstsq(J_aug, r_aug, rcond=None)
        pose = se2_compose((float(delta[1]), float(delta[2]), float(delta[0])), pose)

        fitness_mse = float(np.mean(d_g**2))
        if prev_mse is not None and abs(prev_mse - fitness_mse) < tol: break
        prev_mse = fitness_mse

    return pose, float(prev_mse if prev_mse is not None else 1e9)

def scan_match_2d(prev_pts, cur_pts, init, max_iter=20, trim=0.7, max_corr=0.5, yaw_offsets_deg=(0,), consistency=(0.8, 0.8, 0.45)):
    best_pose, best_mse = init, 1e18

    for off in yaw_offsets_deg:
        cand_init = se2_compose((0.0, 0.0, np.deg2rad(off)), init)
        est, mse = icp_point_to_line_2d(prev_pts, cur_pts, init=cand_init, max_iter=max_iter, trim=trim, max_corr=max_corr)
        if mse < best_mse: best_pose, best_mse = est, mse

    dx, dy, dth = best_pose
    dx_i, dy_i, dth_i = init
    dx_max, dy_max, dth_max = consistency
    if (abs(dx - dx_i) > dx_max) or (abs(dy - dy_i) > dy_max) or (abs(wrap_pi(dth - dth_i)) > dth_max):
        return init, 1e9

    return best_pose, best_mse

def part2_scan_matching(hok, t_enc, poses_center_from_odom, step=1, max_scans=None, rmin=0.2, rmax=20.0, scan_stride=2, icp_max_iter=20, icp_trim=0.7, max_corr=0.5):
    t_l = np.asarray(hok["time_stamps"]).reshape(-1)
    scan_ids = list(range(0, min(t_l.shape[0], max_scans or t_l.shape[0]), step))
    if len(scan_ids) < 2: raise RuntimeError("Not enough scans.")

    traj = [nearest_pose_by_time(t_l[scan_ids[0]], t_enc, poses_center_from_odom)]
    edges, mses = [], []
    t_cl = np.array([LIDAR_X_FROM_CENTER, LIDAR_Y_FROM_CENTER])

    for n in range(1, len(scan_ids)):
        s_prev, s_cur = scan_ids[n-1], scan_ids[n]
        init = se2_relative(
            nearest_pose_by_time(t_l[s_prev], t_enc, poses_center_from_odom),
            nearest_pose_by_time(t_l[s_cur], t_enc, poses_center_from_odom)
        )

        pts_prev = scan_to_points(hok, s_prev, rmin=rmin, rmax=rmax, stride=scan_stride) + t_cl
        pts_cur  = scan_to_points(hok, s_cur,  rmin=rmin, rmax=rmax, stride=scan_stride) + t_cl

        if pts_prev.shape[0] < 80 or pts_cur.shape[0] < 80:
            delta, mse = init, 1e9
        else:
            delta, mse = scan_match_2d(pts_cur, pts_prev, init, max_iter=icp_max_iter, trim=icp_trim, max_corr=max_corr, consistency=(0.8, 0.8, 0.45))

        mses.append(mse)
        traj.append(se2_compose(traj[-1], delta))
        edges.append((len(traj)-2, len(traj)-1, delta[0], delta[1], delta[2], mse, s_prev, s_cur))

    return t_l[scan_ids], np.array(traj, float), edges, np.array(mses, float), np.array(scan_ids, int)

def bresenham2D(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = 1 if x1 >= x0 else -1, 1 if y1 >= y0 else -1
    xs, ys = [], []
    
    if dy <= dx:
        err, y = dx / 2, y0
        for x in range(x0, x1 + sx, sx):
            xs.append(x); ys.append(y)
            err -= dy
            if err < 0: y += sy; err += dx
    else:
        err, x = dy / 2, x0
        for y in range(y0, y1 + sy, sy):
            xs.append(x); ys.append(y)
            err -= dx
            if err < 0: x += sx; err += dy
    return np.array(xs), np.array(ys)

def init_map(res=0.05, xy_min=-25.0, xy_max=25.0):
    size = np.ceil((np.array([xy_max, xy_max]) - np.array([xy_min, xy_min])) / res).astype(int)
    size[size % 2 == 0] += 1
    return {"res": float(res), "min": np.array([xy_min, xy_min], float), "size": size, "logodds": np.zeros(size, float)}

def world_to_cell(MAP, pts_w): return np.floor((pts_w - MAP["min"]) / MAP["res"]).astype(int)
def in_map(MAP, c): return (0 <= c[0] < MAP["size"][0]) and (0 <= c[1] < MAP["size"][1])

def build_occupancy_from_nodes(hok, traj, scan_ids, res=0.05, lo_occ=0.85, lo_free=-0.4, lo_clip=5.0, rmin=0.2, rmax=15.0, scan_stride=1):
    MAP = init_map(res=res)
    t_cl = np.array([LIDAR_X_FROM_CENTER, LIDAR_Y_FROM_CENTER])
    ang = lidar_angles(hok)

    for node_i, scan_id in enumerate(scan_ids):
        ranges = hok["ranges"][:, scan_id].astype(float)
        mask = np.isfinite(ranges) & (ranges > rmin) & (ranges < rmax)
        r, a = ranges[mask], ang[mask]
        
        pts_c = np.stack([r*np.cos(a), r*np.sin(a)], axis=1)[::scan_stride] + t_cl
        pts_w = transform_points(tuple(traj[node_i]), pts_c)

        lidar_cell = world_to_cell(MAP, transform_points(tuple(traj[node_i]), t_cl.reshape(1, 2))[0].reshape(1, 2))[0]

        for p in pts_w:
            end = world_to_cell(MAP, p.reshape(1,2))[0]
            if not in_map(MAP, end) or not in_map(MAP, lidar_cell): continue
            
            xs, ys = bresenham2D(lidar_cell[0], lidar_cell[1], end[0], end[1])
            if xs.size < 2: continue
            
            MAP["logodds"][xs[:-1], ys[:-1]] += lo_free
            MAP["logodds"][xs[-1], ys[-1]] += lo_occ

        MAP["logodds"] = np.clip(MAP["logodds"], -lo_clip, lo_clip)

    return MAP, 1.0 - 1.0 / (1.0 + np.exp(MAP["logodds"]))

def build_texture_map(kin, t_traj, traj, MAP, data_dir, seq):
    if kin is None: return None
    print(f"Building Texture Map from seq {seq}...")
    
    t_disp = np.asarray(kin.get("disparity_time_stamps")).reshape(-1)
    t_rgb  = np.asarray(kin.get("rgb_time_stamps")).reshape(-1)

    tex_map = np.zeros((*MAP["size"], 3), dtype=float)
    color_weight = np.zeros(MAP["size"], dtype=float)

    fsu, fsv, cu, cv = 585.051, 585.051, 242.94, 315.84
    R_c2r = R_scipy.from_euler('ZYX', [0.021, 0.36, 0.0]).as_matrix()
    t_c2r = np.array([0.18, 0.005, 0.36])
    
    u, v = np.meshgrid(np.arange(640), np.arange(480))
    
    for k in range(0, t_disp.shape[0], 5):
        t = t_disp[k]
        pose = traj[np.argmin(np.abs(t_traj - t))]
        rgb_k = np.argmin(np.abs(t_rgb - t))
        
        disp_file = os.path.join(data_dir, "dataRGBD", f"Disparity{seq}", f"disparity{seq}_{k+1}.png")
        rgb_file  = os.path.join(data_dir, "dataRGBD", f"RGB{seq}", f"rgb{seq}_{rgb_k+1}.png")
        
        if not os.path.exists(disp_file) or not os.path.exists(rgb_file): continue
            
        d_img = cv2.imread(disp_file, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB) if cv2.imread(rgb_file) is not None else None
        if d_img is None or img is None: continue
        
        dd = -0.00304 * d_img.astype(float) + 3.31
        depth = np.where(dd > 0.1, 1.03 / dd, 0)
        
        # Valid depth filtering
        valid = (dd > 0.1) & (depth > 0.2) & (depth < 3.0)
        z_opt, u_val, v_val, dd_val = depth[valid], u[valid], v[valid], dd[valid]
        
        pts_cam = np.stack([z_opt, -(u_val - cu) * z_opt / fsu, -(v_val - cv) * z_opt / fsv], axis=1)
        pts_bot = (R_c2r @ pts_cam.T).T + t_c2r
        
        c_yaw, s_yaw = np.cos(pose[2]), np.sin(pose[2])
        pts_world = (np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1]]) @ pts_bot.T).T + np.array([pose[0], pose[1], 0])
        
        # Floor threshold
        floor_mask = (pts_world[:, 2] > -0.08) & (pts_world[:, 2] < 0.08)
        floor_pts = pts_world[floor_mask]
        if floor_pts.shape[0] == 0: continue
            
        u_f, v_f, dd_f = u_val[floor_mask], v_val[floor_mask], dd_val[floor_mask]
        rgbi = np.round((526.37 * u_f + 19276 - 7877.07 * dd_f) / 585.051).astype(int)
        rgbj = np.round((526.37 * v_f + 16662) / 585.051).astype(int)
        
        valid_rgb = (rgbi >= 0) & (rgbi < 640) & (rgbj >= 0) & (rgbj < 480)
        
        cells = world_to_cell(MAP, floor_pts[valid_rgb][:, :2])
        colors = img[rgbj[valid_rgb], rgbi[valid_rgb]].astype(float) / 255.0
        
        in_bounds = (cells[:, 0] >= 0) & (cells[:, 0] < MAP["size"][0]) & (cells[:, 1] >= 0) & (cells[:, 1] < MAP["size"][1])
        cells, colors = cells[in_bounds], colors[in_bounds]
        
        for (cx, cy), color in zip(cells, colors):
            tex_map[cx, cy] += color
            color_weight[cx, cy] += 1.0

    mask = color_weight > 0
    tex_map[mask] /= color_weight[mask][:, None]
    tex_map[~mask] = [0.5, 0.5, 0.5]
    
    return tex_map

def normalize_pc3(pc):
    pc = np.asarray(pc, dtype=float)
    if pc.shape[0] in (3, 4) and pc.shape[1] > 4: pc = pc.T
    pc = pc[:, :3]
    return pc[np.all(np.isfinite(pc), axis=1)]

def yaw_Rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def best_fit_yaw_only(A, B):
    muA, muB = A.mean(axis=0), B.mean(axis=0)
    H = (A - muA)[:, :2].T @ (B - muB)[:, :2]
    U, _, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[1, :] *= -1
        R2 = Vt.T @ U.T
    
    R = yaw_Rz(float(np.arctan2(R2[1,0], R2[0,0])))
    t = muB - R @ muA
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    
    return T, float(np.mean(np.sum(((A @ R.T + t) - B)**2, axis=1)))

def icp_3d_point_to_point(src, dst, T0=None, max_iter=30, trim=0.7, max_corr=0.05):
    src, dst = normalize_pc3(src), normalize_pc3(dst)
    T = np.eye(4) if T0 is None else T0.copy()
    tree = KDTree(dst)
    src_h = np.c_[src, np.ones((src.shape[0], 1))]
    prev = None

    for _ in range(max_iter):
        cur = (T @ src_h.T).T[:, :3]
        d, idx = tree.query(cur)
        good = d < max_corr
        if np.count_nonzero(good) < 50: break
        
        cur, match, d = cur[good], dst[idx[good]], d[good]
        k = max(50, int(trim * cur.shape[0]))
        sel = np.argsort(d)[:k]
        A, B = cur[sel], match[sel]

        muA, muB = A.mean(axis=0), B.mean(axis=0)
        U, _, Vt = np.linalg.svd((A - muA).T @ (B - muB))
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = Vt.T @ U.T

        dT = np.eye(4)
        dT[:3, :3], dT[:3, 3] = R, muB - R @ muA
        T = dT @ T

        mse = float(np.mean(d[sel]**2))
        if prev is not None and abs(prev - mse) < 1e-8: break
        prev = mse

    return T, float(prev if prev is not None else 1e9)

def save_icp_png(source_pc, target_pc, T, out_png):
    src, tgt = normalize_pc3(source_pc), normalize_pc3(target_pc)
    src_t = (T @ np.c_[src, np.ones((src.shape[0],1))].T).T[:, :3]

    def sample(P, n=2000): return P[np.random.choice(P.shape[0], n, replace=False)] if P.shape[0] > n else P
    src_t, tgt = sample(src_t), sample(tgt)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, i, j) in zip(axs, [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]):
        ax.scatter(tgt[:, i], tgt[:, j], s=1, alpha=0.6, label="target")
        ax.scatter(src_t[:, i], src_t[:, j], s=1, alpha=0.6, label="source(T)")
        ax.set_title(name); ax.axis("equal"); ax.grid(True)
    axs[0].legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def warmup_run(icp_warmup_dir, obj_name, yaw_step_deg=30):
    warmup_root = os.path.abspath(os.path.join(icp_warmup_dir, ".."))
    save_dir = os.path.abspath(os.path.join(os.getcwd(), "../outputs"))
    os.makedirs(save_dir, exist_ok=True)
    
    old_cwd = os.getcwd()
    os.chdir(warmup_root)
    try:
        source_pc = read_canonical_model(obj_name)
        for i in range(4):
            target_pc = load_pc(obj_name, i)
            if np.median(np.linalg.norm(target_pc - np.median(target_pc, axis=0), axis=1)) > 5.0:
                target_pc /= 1000.0

            def downsample(P, n=6000):
                P = normalize_pc3(P)
                return P[np.random.choice(P.shape[0], n, replace=False)] if P.shape[0] > n else P

            src, tgt = downsample(source_pc), downsample(target_pc)
            mu_s, mu_t = src.mean(axis=0), tgt.mean(axis=0)
            best_T, best_mse = None, 1e18

            # Discrete Z-axis search followed by 6-DoF ICP refinement
            for yaw_deg in range(-180, 180, yaw_step_deg):
                R = yaw_Rz(np.deg2rad(yaw_deg))
                T0 = np.eye(4)
                T0[:3, :3], T0[:3, 3] = R, mu_t - R @ mu_s
                
                T, mse = icp_3d_point_to_point(src, tgt, T0=T0, max_iter=40, trim=0.7, max_corr=0.1)
                if mse < best_mse: best_mse, best_T = mse, T

            print(f"Warmup {obj_name} pc#{i} best_mse={best_mse:.6f}")
            try: visualize_icp_result(src, tgt, best_T)
            except: pass
            save_icp_png(src, tgt, best_T, os.path.join(save_dir, f"warmup_{obj_name}_pc{i}.png"))
    finally:
        os.chdir(old_cwd)

def try_import_gtsam():
    try:
        import gtsam; from gtsam import Pose2, symbol
        return gtsam, Pose2, symbol
    except: return None, None, None

def loop_candidates_proximity(traj, radius=1.0, min_sep=30):
    xy, tree = traj[:, :2], KDTree(traj[:, :2])
    pairs = set()
    for i in range(xy.shape[0]):
        for j in tree.query_ball_point(xy[i], r=radius):
            if j > i + min_sep: pairs.add((i, j))
    return sorted(pairs)

def build_loop_edges(hok, traj, scan_ids, pairs, rmin=0.2, rmax=20.0, scan_stride=2, icp_max_iter=25, icp_trim=0.7, max_corr=0.6, mse_th=0.0015):
    t_cl = np.array([LIDAR_X_FROM_CENTER, LIDAR_Y_FROM_CENTER])
    edges = []
    for (i, j) in pairs:
        pts_i = scan_to_points(hok, int(scan_ids[i]), rmin=rmin, rmax=rmax, stride=scan_stride) + t_cl
        pts_j = scan_to_points(hok, int(scan_ids[j]), rmin=rmin, rmax=rmax, stride=scan_stride) + t_cl
        if pts_i.shape[0] < 80 or pts_j.shape[0] < 80: continue
            
        init = se2_relative(tuple(traj[i]), tuple(traj[j]))
        delta, mse = scan_match_2d(pts_j, pts_i, init, max_iter=icp_max_iter, trim=icp_trim, max_corr=max_corr, consistency=(1.2, 1.2, 0.55))
        
        # Filter false loops
        if mse < mse_th and abs(delta[0] - init[0]) < 0.5 and abs(delta[1] - init[1]) < 0.5:
            edges.append((i, j, delta[0], delta[1], delta[2], mse))
            
    return edges

def gtsam_optimize(traj_init, odo_edges, loop_edges, prior_sig=(0.01, 0.01, 0.01), odo_sig=(0.01, 0.01, 0.01), loop_sig=(0.05, 0.05, 0.05)):
    gtsam, Pose2, symbol = try_import_gtsam()
    if gtsam is None: raise RuntimeError("Install gtsam: pip install gtsam")

    graph, initial = gtsam.NonlinearFactorGraph(), gtsam.Values()
    priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array(prior_sig))
    odoNoise   = gtsam.noiseModel.Diagonal.Sigmas(np.array(odo_sig))
    loopNoise  = gtsam.noiseModel.Diagonal.Sigmas(np.array(loop_sig))

    graph.add(gtsam.PriorFactorPose2(symbol('x',0), Pose2(*traj_init[0]), priorNoise))

    for i, (x,y,th) in enumerate(traj_init): initial.insert(symbol('x', i), Pose2(float(x), float(y), float(th)))
    for (i,j,dx,dy,dth) in odo_edges: graph.add(gtsam.BetweenFactorPose2(symbol('x',i), symbol('x',j), Pose2(dx,dy,dth), odoNoise))
    for (i,j,dx,dy,dth,_) in loop_edges: graph.add(gtsam.BetweenFactorPose2(symbol('x',i), symbol('x',j), Pose2(dx,dy,dth), loopNoise))

    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("ERROR")
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()

    out = np.zeros_like(traj_init)
    for i in range(traj_init.shape[0]):
        p = result.atPose2(symbol('x', i))
        out[i] = [p.x(), p.y(), p.theta()]
    return out

def auto_crop_plot(map_data_T, pad=20):
    active = np.argwhere(map_data_T != 0.5) if map_data_T.ndim == 2 else np.argwhere(np.any(map_data_T != [0.5, 0.5, 0.5], axis=-1))
    if len(active) > 0:
        ymin, xmin = active.min(axis=0)
        ymax, xmax = active.max(axis=0)
        plt.xlim(xmin - pad, xmax + pad)
        plt.ylim(ymin - pad, ymax + pad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../data")
    ap.add_argument("--save_dir", type=str, default="../outputs")
    ap.add_argument("--seq", type=int, default=20, choices=[20,21])
    ap.add_argument("--part", type=str, default="all", choices=["1","2","3","4","all","warmup"])
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--max_scans", type=int, default=None)
    ap.add_argument("--rmin", type=float, default=0.2); ap.add_argument("--rmax", type=float, default=20.0)
    ap.add_argument("--scan_stride", type=int, default=2)
    ap.add_argument("--icp_max_iter", type=int, default=20); ap.add_argument("--icp_trim", type=float, default=0.7)
    ap.add_argument("--max_corr", type=float, default=0.5)
    ap.add_argument("--loop_interval", type=int, default=10); ap.add_argument("--loop_radius", type=float, default=1.0)
    ap.add_argument("--loop_min_sep", type=int, default=30); ap.add_argument("--loop_mse_th", type=float, default=0.0015)
    ap.add_argument("--icp_warmup_dir", type=str, default="icp_warm_up/data")
    args = ap.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    if args.part == "warmup":
        warmup_run(args.icp_warmup_dir, "drill", yaw_step_deg=30)
        warmup_run(args.icp_warmup_dir, "liq_container", yaw_step_deg=30)
        return

    enc, imu, hok, kin = load_all_npz(args.data_dir, args.seq)
    t_enc, poses_rear, poses_center = part1_odometry(enc, imu)

    if args.part in ["1","all"]:
        plot_traj(poses_center, f"Part1 Odometry seq={args.seq}")
        plt.savefig(os.path.join(args.save_dir, f"p1_traj_seq{args.seq}.png"), dpi=150)

    if args.part in ["2","3","4","all"]:
        t_nodes, traj_icp, edges_icp, mses, scan_ids = part2_scan_matching(
            hok, t_enc, poses_center, step=args.step, max_scans=args.max_scans,
            rmin=args.rmin, rmax=args.rmax, scan_stride=args.scan_stride,
            icp_max_iter=args.icp_max_iter, icp_trim=args.icp_trim, max_corr=args.max_corr
        )

        if args.part in ["2","all"]:
            plot_traj(traj_icp, f"Part2 ICP traj seq={args.seq}")
            plt.savefig(os.path.join(args.save_dir, f"p2_traj_seq{args.seq}.png"), dpi=150)
            plt.figure(); plt.plot(mses); plt.title("ICP fitness per step"); plt.grid(True)
            plt.savefig(os.path.join(args.save_dir, f"p2_fitness_seq{args.seq}.png"), dpi=150)

    if args.part in ["3","all"]:
        MAP1, prob1 = build_occupancy_from_nodes(hok, traj_icp[:1], scan_ids[:1], rmin=args.rmin, rmax=args.rmax)
        plt.figure(); plt.imshow(prob1.T, origin="lower", cmap="gray"); plt.title(f"Part3 Occ FIRST scan seq={args.seq}")
        plt.savefig(os.path.join(args.save_dir, f"p3_occ_first_seq{args.seq}.png"), dpi=150)

        MAP, prob = build_occupancy_from_nodes(hok, traj_icp, scan_ids, rmin=args.rmin, rmax=args.rmax)
        plt.figure(); plt.imshow(prob.T, origin="lower", cmap="gray"); auto_crop_plot(prob.T)
        plt.title(f"Part3 Occ FULL seq={args.seq}")
        plt.savefig(os.path.join(args.save_dir, f"p3_occ_full_seq{args.seq}.png"), dpi=150)

        tex_map = build_texture_map(kin, t_nodes, traj_icp, MAP, args.data_dir, args.seq)
        if tex_map is not None:
            plt.figure(figsize=(10, 10)); tex_map_T = np.transpose(tex_map, (1, 0, 2))
            plt.imshow(tex_map_T, origin="lower"); auto_crop_plot(tex_map_T)
            plt.title(f"Part3 Texture Map seq={args.seq}")
            plt.savefig(os.path.join(args.save_dir, f"p3_texture_seq{args.seq}.png"), dpi=150)

    if args.part in ["4","all"]:
        odo_edges = [(i, j, dx, dy, dth) for (i,j,dx,dy,dth,_,_,_) in edges_icp]
        pairs = [(i, i+args.loop_interval) for i in range(0, traj_icp.shape[0]-args.loop_interval)] + \
                loop_candidates_proximity(traj_icp, radius=args.loop_radius, min_sep=args.loop_min_sep)

        loop_edges = build_loop_edges(hok, traj_icp, scan_ids, pairs, rmin=args.rmin, rmax=args.rmax, 
                                      scan_stride=args.scan_stride, icp_max_iter=max(25, args.icp_max_iter), 
                                      icp_trim=args.icp_trim, max_corr=max(0.6, args.max_corr), mse_th=args.loop_mse_th)

        traj_opt = gtsam_optimize(traj_icp, odo_edges, loop_edges)
        plot_traj(traj_opt, f"Part4 AFTER GTSAM seq={args.seq}")
        plt.savefig(os.path.join(args.save_dir, f"p4_traj_seq{args.seq}.png"), dpi=150)

        MAP2, prob2 = build_occupancy_from_nodes(hok, traj_opt, scan_ids, rmin=args.rmin, rmax=args.rmax)
        plt.figure(); plt.imshow(prob2.T, origin="lower", cmap="gray"); auto_crop_plot(prob2.T)
        plt.title(f"Part4 Occ AFTER GTSAM seq={args.seq}")
        plt.savefig(os.path.join(args.save_dir, f"p4_occ_seq{args.seq}.png"), dpi=150)

        tex_map_opt = build_texture_map(kin, t_nodes, traj_opt, MAP2, args.data_dir, args.seq)
        if tex_map_opt is not None:
            plt.figure(figsize=(10, 10)); tex_map_opt_T = np.transpose(tex_map_opt, (1, 0, 2))
            plt.imshow(tex_map_opt_T, origin="lower"); auto_crop_plot(tex_map_opt_T)
            plt.title(f"Part4 Texture AFTER GTSAM seq={args.seq}")
            plt.savefig(os.path.join(args.save_dir, f"p4_texture_seq{args.seq}.png"), dpi=150)

if __name__ == "__main__":
    main()