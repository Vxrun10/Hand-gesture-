import cv2
import numpy as np
import math
from collections import deque

# ---------------- CONFIG ----------------
CAM_WIDTH = 960
CAM_HEIGHT = 720
MIN_CONTOUR_AREA = 3000
MOTION_HISTORY_LEN = 5         # for smoothing hand center
MOTION_MIN_DIST = 20           # px threshold to consider as movement
# ----------------------------------------


def create_trackbars():
    """Create HSV trackbars so you can tune skin detection for your hand."""
    cv2.namedWindow("Controls")
    cv2.resizeWindow("Controls", 500, 250)

    # Hue: 0–179, Sat: 0–255, Val: 0–255
    cv2.createTrackbar("H_min", "Controls", 0, 179, lambda x: None)
    cv2.createTrackbar("H_max", "Controls", 25, 179, lambda x: None)

    cv2.createTrackbar("S_min", "Controls", 30, 255, lambda x: None)
    cv2.createTrackbar("S_max", "Controls", 255, 255, lambda x: None)

    cv2.createTrackbar("V_min", "Controls", 40, 255, lambda x: None)
    cv2.createTrackbar("V_max", "Controls", 255, 255, lambda x: None)


def get_hsv_range_from_trackbars():
    """Read HSV range from trackbars."""
    h_min = cv2.getTrackbarPos("H_min", "Controls")
    h_max = cv2.getTrackbarPos("H_max", "Controls")
    s_min = cv2.getTrackbarPos("S_min", "Controls")
    s_max = cv2.getTrackbarPos("S_max", "Controls")
    v_min = cv2.getTrackbarPos("V_min", "Controls")
    v_max = cv2.getTrackbarPos("V_max", "Controls")

    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return lower, upper


def get_skin_mask(frame, lower, upper):
    """Return cleaned skin mask using HSV inRange + morphology."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def find_biggest_contour(mask):
    """Find biggest contour above area threshold."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
        return None
    return c


def get_center_of_contour(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def analyze_fingers(cnt, frame):
    """
    Use convex hull + convexity defects to estimate finger count (0–5)
    and fingertip positions.
    """
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return 0, []

    defects = cv2.convexityDefects(cnt, hull_indices)
    if defects is None:
        return 0, []

    finger_points = []

    # Get contour center for reference
    center = get_center_of_contour(cnt)
    if center is None:
        center = (0, 0)
    cx, cy = center

    fingers = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # Calculate the angle between start-far-end
        a = math.dist(end, far)
        b = math.dist(start, far)
        c = math.dist(start, end)

        if b * a == 0:
            continue

        angle = math.degrees(math.acos((b * b + a * a - c * c) / (2 * a * b)))

        # Heuristic: valid finger gap if angle is small and defect is deep enough
        if angle < 80 and d > 10000 and far[1] < cy + 40:
            fingers += 1
            # The tip is roughly the higher of start/end
            tip = start if start[1] < end[1] else end
            finger_points.append(tip)

            cv2.circle(frame, tip, 10, (0, 0, 255), 2)
            cv2.circle(frame, far, 6, (0, 255, 0), -1)
            cv2.line(frame, start, end, (255, 255, 0), 2)

    # number of fingers is usually defects + 1 when hand is open
    if fingers > 0:
        fingers = min(fingers + 1, 5)

    return fingers, finger_points


def classify_hand_pose(finger_count):
    if finger_count == 0:
        return "FIST"
    elif finger_count == 1:
        return "ONE (Pointer)"
    elif finger_count == 2:
        return "TWO"
    elif finger_count == 3:
        return "THREE"
    elif finger_count == 4:
        return "FOUR"
    elif finger_count == 5:
        return "OPEN PALM"
    else:
        return "UNKNOWN"


def classify_motion(history):
    """Classify direction (LEFT/RIGHT/UP/DOWN/STATIC) using last few centers."""
    if len(history) < 2:
        return "STATIC"

    x_coords = [p[0] for p in history]
    y_coords = [p[1] for p in history]

    dx = x_coords[-1] - x_coords[0]
    dy = y_coords[-1] - y_coords[0]

    dist = math.hypot(dx, dy)
    if dist < MOTION_MIN_DIST:
        return "STATIC"

    # Horizontal vs vertical bias
    if abs(dx) > abs(dy):
        return "MOVING RIGHT" if dx > 0 else "MOVING LEFT"
    else:
        return "MOVING DOWN" if dy > 0 else "MOVING UP"


def main():
    create_trackbars()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    center_history = deque(maxlen=MOTION_HISTORY_LEN)
    fps_time = cv2.getTickCount()
    fps = 0.0

    print("Hand analyzer started. Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        lower, upper = get_hsv_range_from_trackbars()
        mask = get_skin_mask(frame, lower, upper)
        cnt = find_biggest_contour(mask)

        pose_text = "No hand"
        motion_text = "STATIC"
        finger_count = 0

        # show small mask preview in top-left
        small_mask = cv2.resize(mask, (320, 240))
        frame[0:240, 0:320] = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)

        if cnt is not None:
            # draw contour and convex hull
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            hull_pts = cv2.convexHull(cnt)
            cv2.polylines(frame, [hull_pts], True, (255, 0, 0), 2)

            center = get_center_of_contour(cnt)
            if center is not None:
                cx, cy = center
                cv2.circle(frame, center, 7, (255, 255, 255), -1)
                center_history.append(center)

                finger_count, finger_tips = analyze_fingers(cnt, frame)
                pose_text = classify_hand_pose(finger_count)
                motion_text = classify_motion(center_history)

                # draw hand bounding box
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 150, 255), 2)

        else:
            center_history.clear()

        # FPS
        new_time = cv2.getTickCount()
        time_diff = (new_time - fps_time) / cv2.getTickFrequency()
        fps = 1.0 / time_diff if time_diff > 0 else fps
        fps_time = new_time

        # Overlay info panel
        cv2.rectangle(frame, (6, 250), (330, 380), (0, 0, 0), -1)
        cv2.putText(frame, f"Pose: {pose_text}", (12, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"Fingers: {finger_count}", (12, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"Motion: {motion_text}", (12, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (12, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Hand Analyzer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
