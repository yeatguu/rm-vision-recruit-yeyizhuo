import cv2
import numpy as np
import glob
import os

# ================= 配置区域 =================
# 棋盘格内角点 (横向, 纵向)
# 格子横10竖7 -> 内角点 (9, 6)
# 格子横7竖10 -> 内角点 (6, 9)
# 先试一个，检测成功数少就换另一个
pattern_size = (9, 6)

folder = r"D:\Pictures\Screenshots"
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 是否只使用内圈角点（去掉最外一圈）
USE_INNER = True
BORDER = 1   # 去掉几圈
# ===========================================

exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
images = []
for ext in exts:
    images.extend(glob.glob(os.path.join(folder, ext)))
images = sorted(set(images))

if not images:
    print(f"未找到图片，请检查路径: {folder}")
    input("按回车键退出...")
    exit()

print(f"共找到 {len(images)} 张图片")
print(f"pattern_size = {pattern_size}\n")

cols, rows = pattern_size

# ---- 完整世界坐标 ----
full_objp = np.zeros((cols * rows, 3), np.float32)
full_objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

# ---- 内圈索引 ----
if USE_INNER:
    inner_indices = []
    for r in range(BORDER, rows - BORDER):
        for c in range(BORDER, cols - BORDER):
            inner_indices.append(r * cols + c)
    inner_indices = np.array(inner_indices, dtype=np.int32)
    objp_use = full_objp[inner_indices]
    print(f"完整角点 {cols*rows} 个，使用内圈 {len(inner_indices)} 个")
else:
    objp_use = full_objp
    print(f"使用完整角点 {cols*rows} 个")

objpoints, imgpoints, good_files, gray_shape = [], [], [], None

for fname in images:
    img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[读取失败] {os.path.basename(fname)}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 优先用 SB 版本，鲁棒性更好；没有就退回普通版
    ret, corners = False, None
    if hasattr(cv2, "findChessboardCornersSB"):
        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)

    if not ret:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not ret:
        print(f"[未检测到角点，跳过] {os.path.basename(fname)}")
        continue

    corners_flat = corners.reshape(-1, 2)

    if USE_INNER:
        corners_use = corners_flat[inner_indices].reshape(-1, 1, 2)
    else:
        corners_use = corners

    objpoints.append(objp_use.copy())
    imgpoints.append(corners_use)
    good_files.append(fname)
    gray_shape = gray.shape[::-1]
    print(f"[成功] {os.path.basename(fname)}")

print(f"\n有效标定图片: {len(objpoints)} / {len(images)}")
if len(objpoints) < 3:
    print("有效图片不足，无法标定。")
    input("按回车键退出...")
    exit()

# ---- 标定 ----
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray_shape, None, None)

print(f"\n===== 标定结果 =====")
print(f"RMS: {ret:.6f}")
print("内参矩阵:\n", mtx)
print("畸变系数:\n", dist.ravel())

# ---- 逐图误差 ----
print("\n===== 每张图误差 =====")
errs = []
for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    per_pt = np.linalg.norm(
        imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
    errs.append(per_pt.mean())
    print(f"图 {i+1:2d} [{os.path.basename(good_files[i])}]: "
          f"mean={per_pt.mean():.3f}, max={per_pt.max():.3f}")

# ---- 可视化 ----
print("\n===== 可视化（任意键切换，ESC 退出）=====")
for i in range(len(objpoints)):
    img = cv2.imdecode(np.fromfile(good_files[i], dtype=np.uint8), cv2.IMREAD_COLOR)
    vis = img.copy()
    pts = imgpoints[i].reshape(-1, 2).astype(int)
    for p in pts:
        cv2.circle(vis, tuple(p), 5, (0, 0, 255), -1)
    cv2.putText(vis, f"img {i+1} pts={len(pts)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Corners Used", vis)
    if (cv2.waitKey(0) & 0xFF) == 27:
        break
cv2.destroyAllWindows()

# ---- 剔除误差最大的图重标 ----
if len(objpoints) >= 5:
    worst = int(np.argmax(errs))
    print(f"\n===== 剔除误差最大的图 {worst+1} 后重标 =====")
    keep = [i for i in range(len(objpoints)) if i != worst]
    obj2 = [objpoints[i] for i in keep]
    img2 = [imgpoints[i] for i in keep]
    ret2, mtx2, dist2, _, _ = cv2.calibrateCamera(
        obj2, img2, gray_shape, None, None)
    print(f"剔除后 RMS: {ret2:.6f}")
    print("剔除后内参:\n", mtx2)
    print("剔除后畸变:\n", dist2.ravel())
    if ret2 < ret:
        print(">>> 剔除后更优，建议用这组参数。")
    else:
        print(">>> 剔除后未改善。")

input("\n按回车键退出...")