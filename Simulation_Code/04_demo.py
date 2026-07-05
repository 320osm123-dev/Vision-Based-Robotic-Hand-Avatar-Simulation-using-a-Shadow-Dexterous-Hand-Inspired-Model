import csv
import cv2
import time
import math
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = "hand_landmarker.task"
MUJOCO_MODEL_PATH = r"C:\hand_mujoco_project\scene_right.xml"


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


FINGER_LANDMARKS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}



FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
EVAL_FINGERS = ["index", "middle", "ring", "pinky"]

FINGERTIP_LANDMARKS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}


# MuJoCo Shadow Hand 모델에서 손끝 위치를 찾기 위한 후보 이름
# 모델 XML마다 site/body 이름이 조금 다를 수 있어서 여러 후보를 넣어둠
ROBOT_TIP_CANDIDATES = {
    "thumb": ["rh_thtip", "rh_THtip", "thtip", "THtip", "rh_thdistal", "thdistal"],
    "index": ["rh_fftip", "rh_FFtip", "fftip", "FFtip", "rh_ffdistal", "ffdistal"],
    "middle": ["rh_mftip", "rh_MFtip", "mftip", "MFtip", "rh_mfdistal", "mfdistal"],
    "ring": ["rh_rftip", "rh_RFtip", "rftip", "RFtip", "rh_rfdistal", "rfdistal"],
    "pinky": ["rh_lftip", "rh_LFtip", "lftip", "LFtip", "rh_lfdistal", "lfdistal"],
}


ROBOT_PALM_CANDIDATES = [
    "rh_palm",
    "palm",
    "rh_wrist",
    "wrist",
    "rh_forearm",
    "forearm",
]


FINGER_ACTUATORS = {
    "index": ["rh_A_FFJ3", "rh_A_FFJ0"],
    "middle": ["rh_A_MFJ3", "rh_A_MFJ0"],
    "ring": ["rh_A_RFJ3", "rh_A_RFJ0"],
    "pinky": ["rh_A_LFJ3", "rh_A_LFJ0"],
}


SIDE_ACTUATORS = {
    "index": "rh_A_FFJ4",
    "middle": "rh_A_MFJ4",
    "ring": "rh_A_RFJ4",
    "pinky": "rh_A_LFJ4",
}


SIDE_SIGN = {
    "index": 1.0,
    "middle": 1.0,
    "ring": -1.0,
    "pinky": -1.0,
}


SIDE_SCALE = {
    "index": 1.20,
    "middle": 1.00,
    "ring": 1.00,
    "pinky": 1.20,
}


THUMB_ACTUATORS = {
    "th5": "rh_A_THJ5",
    "th4": "rh_A_THJ4",
    "th3": "rh_A_THJ3",
    "th2": "rh_A_THJ2",
    "th1": "rh_A_THJ1",
}


WRIST_ACTUATORS = {
    "wrist_1": "rh_A_WRJ1",  # 앞뒤 굽힘, 지금은 고정
    "wrist_2": "rh_A_WRJ2",  # 좌우 흔들기
}


# 고정 actuator 없음
NEUTRAL_ACTUATORS = []


# 손목 좌우 움직임 강도
WRIST_LR_SCALE = 1.00

# 손목 좌우 계산용 파라미터
WRIST_LR_MAX_ANGLE_DEG = 25.0
WRIST_LR_DEADZONE = 0.03


def angle_3points(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denom < 1e-8:
        return 180.0

    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)

    return math.degrees(math.acos(cosang))


def calc_bend(hand_landmarks, finger_name):
    ids = FINGER_LANDMARKS[finger_name]

    pts = []

    for idx in ids:
        lm = hand_landmarks[idx]
        pts.append([lm.x, lm.y, lm.z])

    if finger_name == "thumb":
        angle1 = angle_3points(pts[0], pts[1], pts[2])
        angle2 = angle_3points(pts[1], pts[2], pts[3])
        angle_avg = 0.5 * angle1 + 0.5 * angle2

        bend = (165.0 - angle_avg) / 65.0
        bend = bend * 3.0

    else:
        angle_pip = angle_3points(pts[0], pts[1], pts[2])
        angle_dip = angle_3points(pts[1], pts[2], pts[3])
        angle_avg = 0.6 * angle_pip + 0.4 * angle_dip
        bend = (170.0 - angle_avg) / 90.0

    bend = float(np.clip(bend, 0.0, 1.0))

    if finger_name == "index":
        bend = bend * 2.8
        bend = float(np.clip(bend, 0.0, 1.0))

    return bend


def calc_side(hand_landmarks, finger_name):
    if finger_name == "index":
        mcp_id = 5
        pip_id = 6
    elif finger_name == "middle":
        mcp_id = 9
        pip_id = 10
    elif finger_name == "ring":
        mcp_id = 13
        pip_id = 14
    elif finger_name == "pinky":
        mcp_id = 17
        pip_id = 18
    else:
        return 0.0

    mcp = hand_landmarks[mcp_id]
    pip = hand_landmarks[pip_id]

    index_mcp = hand_landmarks[5]
    pinky_mcp = hand_landmarks[17]

    palm_width = abs(index_mcp.x - pinky_mcp.x)

    if palm_width < 1e-5:
        palm_width = 0.1

    side = (pip.x - mcp.x) / palm_width

    deadzone = 0.02

    if abs(side) < deadzone:
        side = 0.0

    side = side * 2.0

    return float(np.clip(side, -1.0, 1.0))


def calc_wrist_lr(hand_landmarks):
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]

    # 손목 -> 중지 MCP 방향 벡터의 기울기 각도로 좌우 계산
    vx = middle_mcp.x - wrist.x
    vy = middle_mcp.y - wrist.y

    lr_angle_deg = math.degrees(math.atan2(vx, -vy))
    wrist_lr = lr_angle_deg / WRIST_LR_MAX_ANGLE_DEG

    return float(np.clip(wrist_lr, -1.0, 1.0))


def actuator_id(model, name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )


def set_actuator_fraction(model, data, name, fraction):
    aid = actuator_id(model, name)

    if aid < 0:
        return

    lo, hi = model.actuator_ctrlrange[aid]

    fraction = float(np.clip(fraction, 0.0, 1.0))
    data.ctrl[aid] = lo + fraction * (hi - lo)


def set_actuator_value(model, data, name, value):
    aid = actuator_id(model, name)

    if aid < 0:
        return

    lo, hi = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = float(np.clip(value, lo, hi))


def set_neutral(model, data, name):
    aid = actuator_id(model, name)

    if aid < 0:
        return

    lo, hi = model.actuator_ctrlrange[aid]

    if lo <= 0.0 <= hi:
        data.ctrl[aid] = 0.0
    else:
        data.ctrl[aid] = 0.5 * (lo + hi)


def make_fingertip_csv_header():
    header = [
        "time_s",
        "alpha",
        "thumb_raw",
        "index_raw",
        "middle_raw",
        "ring_raw",
        "pinky_raw",
    ]

    for prefix in ["human", "robot"]:
        for finger in FINGERS:
            header += [
                f"{prefix}_{finger}_x",
                f"{prefix}_{finger}_y",
                f"{prefix}_{finger}_z",
            ]

    for finger in FINGERS:
        header.append(f"{finger}_tip_error")

    header.append("overall_tip_error")

    return header


def make_raw_landmark_csv_header():
    header = ["time_s"]

    for i in range(21):
        header += [
            f"lm{i}_x",
            f"lm{i}_y",
            f"lm{i}_z",
        ]

    return header


def make_raw_landmark_csv_row(t, hand_landmarks):
    row = [t]

    for lm in hand_landmarks:
        row += [lm.x, lm.y, lm.z]

    return row


def landmark_to_np(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def calc_human_fingertips_norm(hand_landmarks):
    wrist = landmark_to_np(hand_landmarks[0])
    index_mcp = landmark_to_np(hand_landmarks[5])
    pinky_mcp = landmark_to_np(hand_landmarks[17])

    human_scale = np.linalg.norm(index_mcp - pinky_mcp)

    if human_scale < 1e-8:
        human_scale = 1.0

    human_tip_norm = {}

    for finger in FINGERS:
        tip_id = FINGERTIP_LANDMARKS[finger]
        tip = landmark_to_np(hand_landmarks[tip_id])
        human_tip_norm[finger] = (tip - wrist) / human_scale

    return human_tip_norm


def find_mujoco_obj(model, candidates):
    for name in candidates:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name
        )

        if site_id >= 0:
            return ("site", site_id, name)

    for name in candidates:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name
        )

        if body_id >= 0:
            return ("body", body_id, name)

    return None


def get_mujoco_obj_pos(data, obj_info):
    obj_type, obj_id, _ = obj_info

    if obj_type == "site":
        return data.site_xpos[obj_id].copy()

    if obj_type == "body":
        return data.xpos[obj_id].copy()

    return None


def resolve_robot_fingertip_objects(model):
    robot_tip_objs = {}

    print("\nRobot fingertip object check")

    for finger in FINGERS:
        obj_info = find_mujoco_obj(model, ROBOT_TIP_CANDIDATES[finger])
        robot_tip_objs[finger] = obj_info

        if obj_info is None:
            print(f"{finger}: not found")
        else:
            obj_type, obj_id, obj_name = obj_info
            print(f"{finger}: {obj_type} '{obj_name}' id = {obj_id}")

    palm_obj = find_mujoco_obj(model, ROBOT_PALM_CANDIDATES)

    if palm_obj is None:
        print("palm: not found -> fingertip mean position will be used as palm reference")
    else:
        obj_type, obj_id, obj_name = palm_obj
        print(f"palm: {obj_type} '{obj_name}' id = {obj_id}")

    return robot_tip_objs, palm_obj


def calc_robot_fingertips_norm(data, robot_tip_objs, palm_obj, robot_scale):
    robot_tip_pos = {}

    for finger in FINGERS:
        obj_info = robot_tip_objs.get(finger)

        if obj_info is None:
            return None, robot_scale

        pos = get_mujoco_obj_pos(data, obj_info)

        if pos is None:
            return None, robot_scale

        robot_tip_pos[finger] = pos

    if palm_obj is not None:
        palm_pos = get_mujoco_obj_pos(data, palm_obj)
    else:
        palm_pos = np.mean(
            np.vstack([robot_tip_pos[finger] for finger in FINGERS]),
            axis=0
        )

    if palm_pos is None:
        return None, robot_scale

    if robot_scale is None:
        scale = np.linalg.norm(robot_tip_pos["index"] - robot_tip_pos["pinky"])

        if scale < 1e-8:
            scale = 1.0

        robot_scale = scale

    robot_tip_norm = {}

    for finger in FINGERS:
        robot_tip_norm[finger] = (robot_tip_pos[finger] - palm_pos) / robot_scale

    return robot_tip_norm, robot_scale


def calc_fingertip_errors(human_tip_norm, robot_tip_norm, human_ref, robot_ref):
    errors = {}

    for finger in FINGERS:
        human_move = human_tip_norm[finger] - human_ref[finger]
        robot_move = robot_tip_norm[finger] - robot_ref[finger]
        errors[finger] = float(np.linalg.norm(human_move - robot_move))

    overall_error = float(np.mean([errors[finger] for finger in EVAL_FINGERS]))

    return errors, overall_error


def make_fingertip_csv_row(
    t,
    alpha,
    raw_bends,
    human_tip_norm,
    robot_tip_norm,
    tip_errors,
    overall_error
):
    row = [
        t,
        alpha,
        raw_bends["thumb"],
        raw_bends["index"],
        raw_bends["middle"],
        raw_bends["ring"],
        raw_bends["pinky"],
    ]

    for source in [human_tip_norm, robot_tip_norm]:
        for finger in FINGERS:
            row += [
                source[finger][0],
                source[finger][1],
                source[finger][2],
            ]

    for finger in FINGERS:
        row.append(tip_errors[finger])

    row.append(overall_error)

    return row



options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


cap = cv2.VideoCapture(0)

# 랙 줄이기용 카메라 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()


model = mujoco.MjModel.from_xml_path(MUJOCO_MODEL_PATH)
data = mujoco.MjData(model)

print("Shadow Hand loaded.")
print("Actuator count:", model.nu)


# 손목 actuator 확인용 출력
print("\nWrist actuator check")
for name in WRIST_ACTUATORS.values():
    aid = actuator_id(model, name)
    print(name, "id =", aid)

    if aid >= 0:
        print("ctrlrange =", model.actuator_ctrlrange[aid])


robot_tip_objs, robot_palm_obj = resolve_robot_fingertip_objects(model)
robot_scale = None
human_tip_ref = None
robot_tip_ref = None


bends = {
    "thumb": 0.0,
    "index": 0.0,
    "middle": 0.0,
    "ring": 0.0,
    "pinky": 0.0,
}


raw_bends = {
    "thumb": 0.0,
    "index": 0.0,
    "middle": 0.0,
    "ring": 0.0,
    "pinky": 0.0,
}


sides = {
    "index": 0.0,
    "middle": 0.0,
    "ring": 0.0,
    "pinky": 0.0,
}


# 손목 좌우값
wrist_lr_smooth = 0.0

# 처음 손 자세를 좌우 중립 기준으로 저장
wrist_lr_zero = None


alpha = 0.6

log_file = open("fingertip_position_log_timesync_demo_alpha08.csv", "w", newline="", encoding="utf-8")
csv_writer = csv.writer(log_file)
csv_writer.writerow(make_fingertip_csv_header())

raw_log_file = open("raw_hand_landmarks_timesync_demo.csv", "w", newline="", encoding="utf-8")
raw_csv_writer = csv.writer(raw_log_file)
raw_csv_writer.writerow(make_raw_landmark_csv_header())

start_time = time.time()

# 실제 시간과 MuJoCo 시뮬레이션 시간을 동기화
# 루프 1회당 mj_step 1번으로 고정하지 않고, 실제 지난 시간만큼 MuJoCo step을 누적 실행한다.
prev_wall_time = time.time()
sim_time_buffer = 0.0
mj_dt = float(model.opt.timestep)

# 랙 줄이기 핵심
# 2면 2프레임마다 MediaPipe 1번 실행
# 더 버벅이면 3으로 바꾸면 됨
DETECT_EVERY_N_FRAMES = 2
frame_count = 0
last_result = None
robot_log_warning_printed = False


with HandLandmarker.create_from_options(options) as landmarker:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            current_human_tip_norm = None

            ret, frame = cap.read()

            if not ret:
                print("카메라 프레임을 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            frame_count += 1

            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                last_result = result
            else:
                result = last_result

            if result is not None and result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                points = []

                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    z = lm.z
                    points.append((x, y, z))
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                for start, end in HAND_CONNECTIONS:
                    x1, y1, _ = points[start]
                    x2, y2, _ = points[end]
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                t_raw = time.time() - start_time
                raw_csv_writer.writerow(
                    make_raw_landmark_csv_row(
                        t_raw,
                        hand_landmarks
                    )
                )

                for finger in bends.keys():
                    raw_bend = calc_bend(hand_landmarks, finger)
                    raw_bends[finger] = raw_bend
                    bends[finger] = (1.0 - alpha) * bends[finger] + alpha * raw_bend

                current_human_tip_norm = calc_human_fingertips_norm(hand_landmarks)

                for finger in sides.keys():
                    raw_side = calc_side(hand_landmarks, finger)
                    sides[finger] = (1.0 - alpha) * sides[finger] + alpha * raw_side

                # 손목 좌우값 계산
                raw_wrist_lr = calc_wrist_lr(hand_landmarks)

                # 처음 잡힌 손 자세를 중립 기준으로 설정
                if wrist_lr_zero is None:
                    wrist_lr_zero = raw_wrist_lr

                raw_wrist_lr = raw_wrist_lr - wrist_lr_zero

                # 작은 떨림 제거
                if abs(raw_wrist_lr) < WRIST_LR_DEADZONE:
                    raw_wrist_lr = 0.0

                # 좌우 움직임 증폭
                raw_wrist_lr = float(np.clip(raw_wrist_lr * 1.6, -1.0, 1.0))

                wrist_lr_smooth = (1.0 - alpha) * wrist_lr_smooth + alpha * raw_wrist_lr

            for name in NEUTRAL_ACTUATORS:
                set_neutral(model, data, name)

            for finger, acts in FINGER_ACTUATORS.items():
                bend = bends[finger]

                mcp_act = acts[0]
                pip_dip_act = acts[1]

                if finger == "index":
                    set_actuator_fraction(model, data, mcp_act, 1.20 * bend)
                    set_actuator_fraction(model, data, pip_dip_act, 1.50 * bend)
                else:
                    set_actuator_fraction(model, data, mcp_act, 0.75 * bend)
                    set_actuator_fraction(model, data, pip_dip_act, bend)

            for finger, side_act in SIDE_ACTUATORS.items():
                side_value = sides[finger]
                side_value = side_value * SIDE_SIGN[finger] * SIDE_SCALE[finger]
                set_actuator_value(model, data, side_act, side_value)

            thumb = bends["thumb"]
            thumb = float(np.clip(thumb * 1.8, 0.0, 1.0))

            set_actuator_value(model, data, THUMB_ACTUATORS["th5"], 0.65)
            set_actuator_fraction(model, data, THUMB_ACTUATORS["th4"], 0.20 + 0.80 * thumb)
            set_actuator_fraction(model, data, THUMB_ACTUATORS["th3"], 0.50 + 0.45 * thumb)
            set_actuator_fraction(model, data, THUMB_ACTUATORS["th2"], 0.15 + 0.85 * thumb)
            set_actuator_fraction(model, data, THUMB_ACTUATORS["th1"], 0.10 + 0.90 * thumb)

            # 손목 앞뒤는 완전 고정
            set_actuator_value(model, data, WRIST_ACTUATORS["wrist_1"], 0.0)

            # 손목 좌우만 적용
            # 현재 좌우가 반대로 움직였으므로 마이너스 적용
            set_actuator_value(model, data, WRIST_ACTUATORS["wrist_2"], -WRIST_LR_SCALE * wrist_lr_smooth)

            now_wall_time = time.time()
            dt_wall = max(0.0, now_wall_time - prev_wall_time)
            prev_wall_time = now_wall_time

            sim_time_buffer += dt_wall

            while sim_time_buffer >= mj_dt:
                mujoco.mj_step(model, data)
                sim_time_buffer -= mj_dt

            viewer.sync()

            if current_human_tip_norm is not None:
                robot_tip_norm, robot_scale = calc_robot_fingertips_norm(
                    data,
                    robot_tip_objs,
                    robot_palm_obj,
                    robot_scale
                )

                if robot_tip_norm is not None:
                    if human_tip_ref is None:
                        human_tip_ref = {
                            finger: current_human_tip_norm[finger].copy()
                            for finger in FINGERS
                        }

                    if robot_tip_ref is None:
                        robot_tip_ref = {
                            finger: robot_tip_norm[finger].copy()
                            for finger in FINGERS
                        }

                    tip_errors, overall_tip_error = calc_fingertip_errors(
                        current_human_tip_norm,
                        robot_tip_norm,
                        human_tip_ref,
                        robot_tip_ref
                    )

                    t = time.time() - start_time

                    csv_writer.writerow(
                        make_fingertip_csv_row(
                            t,
                            alpha,
                            raw_bends,
                            current_human_tip_norm,
                            robot_tip_norm,
                            tip_errors,
                            overall_tip_error
                        )
                    )
                else:
                    if not robot_log_warning_printed:
                        print("로봇 손끝 site/body를 찾지 못해 fingertip position log를 저장하지 못했습니다.")
                        print("위에 출력된 Robot fingertip object check에서 not found 항목을 확인하세요.")
                        robot_log_warning_printed = True

            cv2.putText(
                frame,
                f"T:{bends['thumb']:.2f} I:{bends['index']:.2f} M:{bends['middle']:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"R:{bends['ring']:.2f} P:{bends['pinky']:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Side I:{sides['index']:.2f} M:{sides['middle']:.2f} R:{sides['ring']:.2f} P:{sides['pinky']:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Wrist LR:{wrist_lr_smooth:.2f} FB:fixed",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Detect every {DETECT_EVERY_N_FRAMES} frames | Press R to recalibrate",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.imshow("MediaPipe Hand Landmarker", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord('r'):
                wrist_lr_zero = None
                wrist_lr_smooth = 0.0
                print("손목 좌우 중립 기준을 다시 잡았습니다.")

log_file.close()
cap.release()
cv2.destroyAllWindows()