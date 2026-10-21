import csv
import numpy as np
import matplotlib.pyplot as plt

# 中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 路径 =====
OBS_PATH   = r"E:\trian\data\kf\track_strafe.csv"
TRUTH_PATH = r"E:\trian\data\kf\track_strafe_truth_first10s.csv"
OUT_CSV    = r"E:\trian\pred_strafe.csv"
OUT_PNG    = r"E:\trian\strafe_compare.png"
DT_PREDICT = 0.25

def load_csv(path, cols):
    data = {c: [] for c in cols}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for c in cols:
                data[c].append(float(row[c]))
    return data

class KF1D:
    def __init__(self, q=1e-2, r=1e-4, init_pos=0.0, init_vel=0.0):
        self.Q = np.eye(2) * q
        self.R = np.array([[r]])
        self.H = np.array([[1.0, 0.0]])
        # 速度初始化归零
        self.x = np.array([[init_pos], [init_vel]])
        # P 矩阵：位置稍大，速度非常大（因为不知道初速度），让滤波器更快信任观测
        self.P = np.diag([10.0, 100.0]) 
        self.A = np.array([[1.0, 1.0 / 30.0],
                           [0.0, 1.0]])

    def set_dt(self, dt):
        self.A = np.array([[1.0, dt],
                           [0.0, 1.0]])

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        # 计算新息（残差）
        y = np.array([[z]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        
        # === 野值剔除 (Outlier Rejection) ===
        # 计算马氏距离的平方 (Mahalanobis distance squared)
        # 如果残差过大，说明观测是野值，直接丢弃，只保留预测值
        d2 = float((y.T @ np.linalg.inv(S) @ y)[0, 0])
        if d2 > 9.0:  # 3-sigma 阈值，可根据情况调整，9.0 代表 3 倍标准差
            # print(f"Rejected outlier: {z}")
            return False  # 返回 False 代表拒绝了观测
            
        # 正常卡尔曼更新
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return True

    def future(self, dt):
        return self.x[0, 0] + self.x[1, 0] * dt

def main():
    obs = load_csv(OBS_PATH, ["t", "x", "y", "z"])
    truth = load_csv(TRUTH_PATH, ["t", "armor_x_future", "armor_y_future", "armor_z_future"])

    ts = np.array(obs["t"])
    if len(ts) == 0:
        raise SystemExit("观测数据为空")

    # 初始速度全部归零
    kf = {
        # x 轴：平移速度变化较快，Q 稍微大一点
        "x": KF1D(q=1e-2, r=1e-3, init_pos=obs["x"][0], init_vel=0.0),
        # y 轴：真值恒定，Q 极小，R 适中
        "y": KF1D(q=1e-4, r=1e-3, init_pos=0.1,         init_vel=0.0),
        # z 轴：距离变化平缓，Q 可以小一点
        "z": KF1D(q=1e-2, r=1e-3, init_pos=obs["z"][0], init_vel=0.0),
    }

    pred_t, pred_x, pred_y, pred_z = [], [], [], []
    prev_t = ts[0]
    
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z"])
        
        for i in range(len(ts)):
            dt = ts[i] - prev_t if i > 0 else 1.0 / 30.0
            prev_t = ts[i]
            
            for axis in ("x", "y", "z"):
                kf[axis].set_dt(dt)
                kf[axis].predict()
                # update 里面会自动判断是否抛弃野值
                kf[axis].update(obs[axis][i])

            fx = kf["x"].future(DT_PREDICT)
            fy = kf["y"].future(DT_PREDICT)
            fz = kf["z"].future(DT_PREDICT)
            
            writer.writerow([f"{ts[i]:.5f}", f"{fx:.4f}", f"{fy:.4f}", f"{fz:.4f}"])
            pred_t.append(ts[i]); pred_x.append(fx); pred_y.append(fy); pred_z.append(fz)

    print(f"[1/3] 已写出 {OUT_CSV}，共 {len(pred_t)} 行")

    # ===== 对比前 10 s 真值 =====
    t_truth = np.array(truth["t"])
    gx = np.array(truth["armor_x_future"])
    gy = np.array(truth["armor_y_future"])
    gz = np.array(truth["armor_z_future"])

    pred_t_arr = np.array(pred_t)
    mask = pred_t_arr <= t_truth[-1] + 1e-6
    fx_cut = np.array(pred_x)[mask]
    fy_cut = np.array(pred_y)[mask]
    fz_cut = np.array(pred_z)[mask]
    n = min(len(fx_cut), len(gx))

    ex = fx_cut[:n] - gx[:n]
    ey = fy_cut[:n] - gy[:n]
    ez = fz_cut[:n] - gz[:n]
    rmse_x = float(np.sqrt((ex**2).mean()))
    rmse_y = float(np.sqrt((ey**2).mean()))
    rmse_z = float(np.sqrt((ez**2).mean()))
    rmse = float(np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2))
    print(f"[2/3] 前 10 s 预测 RMSE = {rmse:.4f} m")
    print(f"      分项: x={rmse_x:.4f}  y={rmse_y:.4f}  z={rmse_z:.4f}")

    # ===== 绘图 =====
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    keys = ["x", "y", "z"]
    truth_keys = ["armor_x_future", "armor_y_future", "armor_z_future"]
    preds = [pred_x, pred_y, pred_z]
    
    for ax, axis, tkey, pdata in zip(axes, keys, truth_keys, preds):
        ax.plot(ts, obs[axis], ".", ms=3, alpha=0.6, label="观测")
        ax.plot(ts, pdata, "-", lw=1.5, label="预测(0.25 s 后)")
        ax.plot(t_truth, truth[tkey], "--", lw=1.5, label="真值(前 10 s)")
        ax.set_ylabel(f"{axis} (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"平移目标：观测 / 预测 / 真值对比   RMSE={rmse:.4f} m")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[3/3] 对比图已保存：{OUT_PNG}")

    plt.show()

if __name__ == "__main__":
    main()