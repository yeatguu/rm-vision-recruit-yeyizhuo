import cv2
import numpy as np

VIDEO = r"E:\trian\armor.mp4"

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"打不开视频: {VIDEO}")

cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)

# 滑动条初值
cv2.createTrackbar("B_minus_R", "binary", 80, 255, lambda x: None)
cv2.createTrackbar("area_min", "binary", 20, 500, lambda x: None)
cv2.createTrackbar("ratio_min_x10", "binary", 20, 100, lambda x: None)  # 实际值 /10

frame_idx = 0
paused = False
frame = None

def process(img, b_thr, area_min, ratio_min):
    b, g, r = cv2.split(img)
    diff = cv2.subtract(b, r)
    _, binary = cv2.threshold(diff, b_thr, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    vis = img.copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lights = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        if w < 1 or h < 1:
            continue
        long_side, short_side = max(w, h), min(w, h)
        ratio = long_side / short_side
        if ratio < ratio_min:
            continue
        lights.append(rect)

    for rect in lights:
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
    return binary, vis, lights

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_idx += 1
    if frame is None:
        continue

    b_thr = cv2.getTrackbarPos("B_minus_R", "binary")
    area_min = cv2.getTrackbarPos("area_min", "binary")
    ratio_min = cv2.getTrackbarPos("ratio_min_x10", "binary") / 10.0

    binary, vis, _ = process(frame, b_thr, area_min, ratio_min)
    cv2.putText(vis, f"frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("binary", binary)
    cv2.imshow("result", vis)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:          # ESC 退出
        break
    elif key == ord(' '):  # 空格暂停/继续
        paused = not paused
    elif key == ord('s'):  # s 截图
        cv2.imwrite(f"tune_frame_{frame_idx}.png", vis)
        print(f"已保存 tune_frame_{frame_idx}.png")

cap.release()
cv2.destroyAllWindows()