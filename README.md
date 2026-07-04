# Webcam-based Shadow Hand Avatar Simulation

## Project Overview

This project implements a webcam-based Shadow Hand avatar simulation using MediaPipe and MuJoCo. Human hand motion is captured from a webcam, converted into finger and wrist motion values, and mapped to the actuator inputs of the MuJoCo Shadow Hand model.

The objective of this project is to retarget human hand motion to a robot hand with different degrees of freedom and evaluate its tracking performance using fingertip-based metrics.

---

## System Pipeline

```text
Webcam Input
      ↓
MediaPipe Hand Landmark Detection
      ↓
21 Hand Keypoints
      ↓
Finger / Wrist Motion Estimation
      ↓
Alpha Smoothing
      ↓
Shadow Hand Actuator Mapping
      ↓
MuJoCo Simulation
      ↓
Fingertip Tracking Evaluation
```

---

## Tech Stack

| Category | Tool |
|---|---|
| Development Environment | VS Code |
| Language | Python |
| Webcam Processing | OpenCV |
| Hand Landmark Detection | MediaPipe Hand Landmarker |
| Robot Hand Model | Shadow Hand robot E3M5 |
| Simulation Environment | MuJoCo |
| Data Analysis | NumPy, Pandas, Matplotlib |

---

## Key Implementation

### Variable Definition

| Symbol | Code Variable | Description |
|---|---|---|
| `Pi` | `hand_landmarks[i]` | i-th hand keypoint extracted from MediaPipe |
| `xi, yi, zi` | `lm.x, lm.y, lm.z` | Coordinate values of keypoint `Pi` |
| `u` | `v1` | First vector used for joint angle calculation |
| `v` | `v2` | Second vector used for joint angle calculation |
| `θ` | `theta` | Joint angle between vectors `u` and `v` |
| `bend` | `bend` | Normalized finger bending value |
| `PIPx` | `pip.x` | x-coordinate of the PIP keypoint |
| `MCPx` | `mcp.x` | x-coordinate of the MCP keypoint |
| `palm_width` | `palm_width` | Reference width of the palm |
| `side` | `side` | Finger side-spreading value |
| `P0` | `hand_landmarks[0]` | Wrist keypoint |
| `P9` | `hand_landmarks[9]` | Middle finger MCP keypoint |
| `vpalm` | `vpalm` | Direction vector from `P0` to `P9` |
| `vx, vy` | `vpalm[0], vpalm[1]` | x and y components of the palm direction vector |
| `wrist_lr` | `wrist_lr` | Left-right wrist rotation value |
| `α` | `alpha` | Smoothing coefficient |
| `previous_value` | `prev_value` | Value from the previous frame |
| `current_value` | `current_value` | Value from the current frame |
| `filtered_value` | `filtered_value` | Final value after alpha smoothing |
| `lo` | `lo` | Minimum actuator control value |
| `hi` | `hi` | Maximum actuator control value |
| `fraction` | `fraction` | Normalized input ratio |
| `ctrl` | `data.ctrl[i]` | Final actuator control input in MuJoCo |
| `RMSE` | `rmse_all` | Root mean square error of fingertip tracking |
| `delay` | `delay_s` | Response delay between human and robot motion |
| `score` | `score` | Final alpha selection score |

---

## Hand Landmark Extraction

Webcam frames were captured using OpenCV and processed with MediaPipe Hand Landmarker. MediaPipe extracts 21 hand keypoints, including the wrist and finger joints.

Each keypoint is defined as:

```text
Pi = (xi, yi, zi),  i = 0, 1, 2, ..., 20
```

`P0` represents the wrist keypoint, and the remaining keypoints represent the joints of the thumb, index, middle, ring, and pinky fingers.

The extracted keypoints were used as input data for estimating finger bending, finger spreading, and wrist motion.

---

## Finger and Wrist Motion Estimation

Finger bending was calculated using the angle between two adjacent finger bone vectors.

```text
θ = cos⁻¹((u · v) / (|u||v|))
```

The calculated angle was normalized into a bending ratio.

```text
bend = normalized(θ)
```

A bend value close to `0` represents an extended finger, while a value close to `1` represents a bent finger.

Finger side motion was estimated from the lateral displacement between the MCP and PIP keypoints.

```text
side = (PIPx - MCPx) / palm_width
```

Wrist left-right motion was estimated using the direction vector from the wrist keypoint `P0` to the middle finger MCP keypoint `P9`.

```text
vpalm = P9 - P0
wrist_lr = atan2(vx, -vy)
```

Through these calculations, human hand motion was converted into robot-controllable motion values.

---

## Alpha Smoothing

MediaPipe landmark data can contain small frame-to-frame noise. To reduce jitter, alpha smoothing was applied to the calculated motion values.

```text
filtered_value = (1 - α) × previous_value + α × current_value
```

A smaller alpha produces smoother motion but increases response delay. A larger alpha improves responsiveness but becomes more sensitive to landmark noise.

Therefore, different alpha values were tested to select an appropriate smoothing coefficient.

---

## MuJoCo Actuator Mapping

The calculated motion values were mapped to the actuator control range of the Shadow Hand model.

MuJoCo does not directly move the fingertip positions. Instead, each actuator receives a control input through `data.ctrl`.

```text
ctrl = lo + fraction × (hi - lo)
```

Here, `lo` and `hi` are the minimum and maximum control values of each actuator, and `fraction` is the normalized input value.

The mapped control values were applied to the Shadow Hand model in MuJoCo, generating robot hand avatar motion based on the webcam input.

---

## Simulation Result & Analysis

### 1. Input Trajectory

The input trajectory was used to verify whether the human hand motion was correctly captured from the webcam.

Fingertip displacement was calculated from the initial position of each finger. During repeated grasping and opening motion, the displacement increased and decreased periodically.

This confirms that the MediaPipe landmark data properly captured the input hand motion.

<p align="center">
  <img src="timesync_alpha_fingertip_analysis_figures/timesync_01_input_trajectory.png" width="700">
</p>

---

### 2. Tracking Comparison

The tracking comparison graph compares the fingertip displacement of the human hand and the MuJoCo Shadow Hand.

The robot hand displacement changed according to the grasping and opening motion of the human hand. This shows that the MediaPipe landmark-based motion values were successfully converted into Shadow Hand avatar motion.

Since the human hand and Shadow Hand have different joint structures and link lengths, the trajectories are not completely identical. The main evaluation focus was the overall tracking trend rather than perfect position matching.

<p align="center">
  <img src="timesync_alpha_fingertip_analysis_figures/timesync_02_tracking_comparison.png" width="700">
</p>

---

### 3. Fingertip Error

Fingertip error was calculated to quantitatively evaluate the tracking performance of the robot hand.

```text
fingertip error = distance between human fingertip and robot fingertip
```

The error was calculated as the distance between the normalized human fingertip position and the normalized robot fingertip position.

The error increased during fast transition motions, such as grasping and opening, and decreased when the hand posture became stable.

The thumb showed a different error pattern because its opposition and rotation motion are more complex than the other fingers.

<p align="center">
  <img src="timesync_alpha_fingertip_analysis_figures/timesync_03_fingertip_error.png" width="700">
</p>

---

### 4. Alpha Selection

The smoothing coefficient alpha was selected by comparing RMSE and response delay.

RMSE represents the overall fingertip tracking error, while delay represents the response lag between the human hand input and the robot hand motion.

Because RMSE and delay have different units, min-max normalization was applied.

```text
normalized value = (value - min value) / (max value - min value)
```

The final score was calculated using equal weights for normalized RMSE and normalized delay.

```text
score = 0.5 × normalized RMSE + 0.5 × normalized delay
```

A smaller alpha produced smoother motion but increased response delay. A larger alpha improved responsiveness but became more sensitive to landmark noise.

The final alpha value was selected by balancing tracking accuracy and response delay.

<p align="center">
  <img src="timesync_alpha_fingertip_analysis_figures/timesync_04_alpha_selection.png" width="700">
</p>

---

## Result Summary

This project implemented a webcam-based Shadow Hand avatar simulation using MediaPipe and MuJoCo.

Human hand landmarks were extracted from webcam input, and the motion was converted into finger bending, finger spreading, and wrist motion values. These values were mapped to the actuator inputs of the MuJoCo Shadow Hand model.

The tracking performance was evaluated using fingertip error, RMSE, delay, and alpha score. The final smoothing coefficient was selected by considering both tracking accuracy and response delay.

---

## Limitations and Future Work

The human hand and Shadow Hand have different joint structures and degrees of freedom, which makes exact fingertip matching difficult.

Thumb motion is especially difficult to reproduce because it includes opposition and complex rotation. MediaPipe landmark noise can also affect the stability of the robot hand motion.

Future work will focus on improving the motion retargeting accuracy using inverse kinematics or learning-based mapping methods.
