import os
import cv2
import json
import csv
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CAMERA_INFO = os.path.join(BASE, "data", "armor", "camera_info.json")
VIDEO_PATH  = os.path.join(BASE, "data", "armor", "armor.mp4")
CSV_OUT     = os.path.join(BASE, "detections_kf.csv")
VIS_OUT     = os.path.join(BASE, "detections_kf_vis.mp4")

class Mode:
    NORMAL = 0 
    TURBO = 1  

def load_camera(path):
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    if "camera_matrix" in info:
        cm = info["camera_matrix"]
        K = np.array(cm["data"] if isinstance(cm, dict) else cm, dtype=np.float64).reshape(3, 3)
        dc = info.get("distortion_coefficients")
        D = np.array(dc["data"] if isinstance(dc, dict) else dc, dtype=np.float64).reshape(-1, 1) if dc else np.zeros((5,1), dtype=np.float64)
        return K, D
    if "K" in info:
        return np.array(info["K"], dtype=np.float64).reshape(3,3), np.array(info.get("D", [0]*5), dtype=np.float64).reshape(-1,1)
    if "fx" in info:
        K = np.array([[info["fx"], 0, info["cx"]], [0, info["fy"], info["cy"]], [0, 0, 1]], dtype=np.float64)
        D = np.array([info.get("k1",0), info.get("k2",0), info.get("p1",0), info.get("p2",0), info.get("k3",0)], dtype=np.float64).reshape(-1,1)
        return K, D
    raise KeyError(f"无法识别: {list(info.keys())}")

def blue_mask(img_bgr, mode):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 恢复正常的 HSV 阈值
    mask = cv2.inRange(hsv, (90, 60, 60), (140, 255, 255))
    if mode == Mode.NORMAL:
        # 标准模式：细长形态学，保护灯条原本形状
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    else:
        # TURBO 模式：稍微加宽一点连通断裂，但绝不横连
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4)))
    return mask

def bar_endpoints(bar):
    box = cv2.boxPoints(bar["rect"]).astype(np.float64)
    box = box[np.argsort(box[:, 1])]
    return box[:2].mean(axis=0), box[2:].mean(axis=0)

def find_bars(mask, mode, min_area=10):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area: continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), ang = rect
        if w > h: w, h = h, w
        ratio = h / max(w, 1e-6)
        
        if mode == Mode.NORMAL:
            if not (1.4 < ratio < 12 and h > 8): continue
        else:
            if not (1.1 < ratio < 15 and h > 6): continue
            
        bars.append({"center": (cx, cy), "w": w, "h": h, "rect": rect, "area": area})
    return bars

def merge_bars(bars, mode):
    if not bars: return bars
    bars = sorted(bars, key=lambda b: -b["area"])
    merged = []
    for b in bars:
        cx, cy = b["center"]; h = b["h"]
        dup = False
        for m in merged:
            mx, my = m["center"]; mh = m["h"]
            if abs(cx - mx) < 0.6 * max(b["w"], m["w"]) and abs(cy - my) < 0.5 * max(h, mh):
                dup = True; break
        if not dup: merged.append(b)
    return merged

def pair_bars(bars, mode):
    pairs = []
    for i in range(len(bars)):
        for j in range(i + 1, len(bars)):
            a, b = bars[i], bars[j]
            if a["center"][0] > b["center"][0]: a, b = b, a
            dx = b["center"][0] - a["center"][0]
            dy = abs(a["center"][1] - b["center"][1])
            h_avg = (a["h"] + b["h"]) / 2
            
            if mode == Mode.NORMAL:
                if abs(a["h"] - b["h"]) / h_avg > 0.5 or dy > 0.5 * h_avg: continue
                if not (0.8 * h_avg < dx < 3.5 * h_avg): continue
            else:
                # TURBO 模式允许剧烈形变
                if abs(a["h"] - b["h"]) / h_avg > 0.7 or dy > 0.7 * h_avg: continue
                if not (0.5 * h_avg < dx < 4.5 * h_avg): continue

            expected_dx = 2.4 * h_avg
            ratio_penalty = abs(dx - expected_dx) / expected_dx
            score = abs(a["h"]-b["h"])/h_avg + dy/h_avg + ratio_penalty
            pairs.append((score, a, b))
    pairs.sort(key=lambda p: p[0])
    return [(a, b) for _, a, b in pairs]

def check_digit_region(img_bgr, lt, lb, rb, rt, mode):
    """
    【最核心的防噪点逻辑】：
    装甲板两根灯带之间一定有一块浅灰/白色的底板。
    如果中间是黑底、或者完全是背景，直接拒绝。
    """
    quad = np.array([lt, lb, rb, rt], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(quad)
    x, y = max(x, 0), max(y, 0)
    roi = img_bgr[y:y+h, x:x+w]
    if roi.size == 0: return False

    margin_x = int(w * 0.25)
    if w - 2 * margin_x < 5: return False 
    
    roi_inner = roi[:, margin_x:w-margin_x]
    gray = cv2.cvtColor(roi_inner, cv2.COLOR_BGR2GRAY)
    
    # 纯色判定：过滤纯黑（阴影）或纯蓝背景
    gray_mask = (gray > 60) & (gray < 240)
    gray_ratio = np.mean(gray_mask)
    
    if mode == Mode.NORMAL:
        return gray_ratio > 0.5
    else:
        return gray_ratio > 0.3

OBJ_PTS = np.array([
    [-0.0675, -0.028, 0.0], [-0.0675, 0.028, 0.0],
    [ 0.0675,  0.028, 0.0], [ 0.0675, -0.028, 0.0]
], dtype=np.float64)

def solve_distance(img_pts, K, D):
    pts = cv2.undistortPoints(img_pts.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(OBJ_PTS, pts, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok: return None, None, None
    return float(np.linalg.norm(tvec)), rvec, tvec

class KalmanFilter3D:
    def __init__(self):
        self.X = np.zeros((6, 1))
        self.P = np.eye(6) * 500.0
        self.Q = np.eye(6) * 0.05
        self.R = np.eye(3) * 0.05
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1
        self.initialized = False
        self.lost_frames = 0
        self.last_Z = None

    def predict(self, dt):
        F = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + self.Q
        return self.X

    def update(self, Z, mode):
        Z = Z.reshape(3, 1)
        if self.last_Z is not None:
            dz = np.linalg.norm(Z - self.last_Z)
            if dz > 0.3: 
                self.Q = np.eye(6) * 0.5
            else:
                self.Q = np.eye(6) * 0.05
        self.last_Z = Z.copy()
        
        y = Z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        d2 = float((y.T @ np.linalg.inv(S) @ y)[0, 0])
        
        threshold = 25.0 if mode == Mode.NORMAL else 80.0
        if d2 > threshold: return False
        
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.lost_frames = 0
        return True

    def predict_future(self, dt_pred):
        x_future = self.X[0, 0] + self.X[3, 0] * dt_pred
        y_future = self.X[1, 0] + self.X[4, 0] * dt_pred
        z_future = self.X[2, 0] + self.X[5, 0] * dt_pred
        return np.array([x_future, y_future, z_future]), np.linalg.norm([x_future, y_future, z_future])

def detect_frame(img, K, D, kf, dt, mode, predict_time=0.25):
    mask = blue_mask(img, mode)
    bars = merge_bars(find_bars(mask, mode), mode)
    pairs = pair_bars(bars, mode)
    
    raw_dets = []
    for left, right in pairs:
        lt, lb = bar_endpoints(left)
        rt, rb = bar_endpoints(right)
        pts = np.array([lt, lb, rb, rt], dtype=np.float64)
        
        if not check_digit_region(img, lt, lb, rb, rt, mode):
            continue
            
        dist, rvec, tvec = solve_distance(pts, K, D)
        if dist is not None:
            if dist < 0.3: continue
            raw_dets.append({
                "points": pts, "distance": dist, "tvec": tvec.reshape(3),
                "left_bar": left, "right_bar": right
            })
            
    raw_dets.sort(key=lambda d: d["distance"])
    final_dets = []
    used_bars = set()
    for d in raw_dets:
        left_id = (round(d["left_bar"]["center"][0]), round(d["left_bar"]["center"][1]))
        right_id = (round(d["right_bar"]["center"][0]), round(d["right_bar"]["center"][1]))
        if left_id not in used_bars and right_id not in used_bars:
            used_bars.add(left_id)
            used_bars.add(right_id)
            final_dets.append(d)

    if final_dets:
        best_det = final_dets[0]
        obs_vec = best_det["tvec"]
        if kf.lost_frames > 15: kf.initialized = False
        if kf.initialized: kf.predict(dt)
        else:
            kf.X[:3] = obs_vec.reshape(3, 1)
            kf.initialized = True
        kf.update(obs_vec, mode)
        future_pos, future_dist = kf.predict_future(predict_time)
        for d in final_dets:
            d["filtered_dist"] = float(np.linalg.norm(kf.X[:3]))
            d["future_dist"] = future_dist
            d["future_pos"] = future_pos
    else:
        kf.lost_frames += 1
            
    return final_dets, kf.lost_frames

def main():
    K, D = load_camera(CAMERA_INFO)
    cap = cv2.VideoCapture(VIDEO_PATH)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dt = 1.0 / fps
    out = cv2.VideoWriter(VIS_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    kf = KalmanFilter3D()

    current_mode = Mode.NORMAL
    lost_counter = 0
    switch_cooldown = 0

    with open(CSV_OUT, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame","lt_x","lt_y","lb_x","lb_y","rb_x","rb_y","rt_x","rt_y","raw_distance","filtered_distance","pred_dist_0.25s","mode"])
        idx = 0; total_det = 0
        
        while True:
            ok, frame = cap.read()
            if not ok: break
            
            if switch_cooldown > 0:
                switch_cooldown -= 1
                
            if current_mode == Mode.NORMAL and lost_counter > 3 and switch_cooldown == 0:
                current_mode = Mode.TURBO
                switch_cooldown = 30
                print(f"帧 {idx}: 切换到 TURBO 模式")
                
            elif current_mode == Mode.TURBO and lost_counter == 0 and switch_cooldown == 0:
                current_mode = Mode.NORMAL
                switch_cooldown = 30
                print(f"帧 {idx}: 切回 NORMAL 模式")

            dets, kf_lost = detect_frame(frame, K, D, kf, dt, current_mode)
            lost_counter = kf_lost
            
            vis = frame.copy()
            for d in dets:
                p = d["points"].astype(int)
                for k in range(4):
                    cv2.line(vis, tuple(p[k]), tuple(p[(k+1)%4]), (0,255,0), 2)
                
                raw_dist = d["distance"]
                if "filtered_dist" in d:
                    f_dist = d["filtered_dist"]
                    p_dist = d["future_dist"]
                    text = f"Raw:{raw_dist:.2f}m Filt:{f_dist:.2f}m Pred:{p_dist:.2f}m"
                    cv2.putText(vis, text, (p[0][0], p[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    cv2.putText(vis, f"{raw_dist:.2f}m", tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
                flat = d["points"].reshape(-1)
                wcsv.writerow([idx] + [f"{v:.1f}" for v in flat] + [f"{d['distance']:.3f}"] + 
                              ([f"{d['filtered_dist']:.3f}", f"{d['future_dist']:.3f}"] if "filtered_dist" in d else ["", ""]) +
                              [current_mode])
                total_det += 1
                
            mode_text = "NORMAL" if current_mode == Mode.NORMAL else "TURBO"
            cv2.putText(vis, f"Mode: {mode_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            
            out.write(vis)
            idx += 1
            if idx % 60 == 0: print(f"处理 {idx} 帧, 累计检测 {total_det}")
            
    cap.release(); out.release()
    print(f"完成！输出至: {CSV_OUT}, {VIS_OUT}")

if __name__ == "__main__":
    main()