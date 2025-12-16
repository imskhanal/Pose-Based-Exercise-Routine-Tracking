import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
from pupil_apriltags import Detector as AprilDetector

APRILTAG_AVAILABLE = True
APRILTAG_TYPE = "pupil"

# try:
#     import apriltag
#     APRILTAG_AVAILABLE = True
#     APRILTAG_TYPE = "apriltag"
# except ImportError:
#     try:
#         from pupil_apriltags import Detector as AprilDetector
#         APRILTAG_AVAILABLE = True
#         APRILTAG_TYPE = "pupil"
#     except ImportError:
#         APRILTAG_AVAILABLE = False
#         APRILTAG_TYPE = None

TAG_WEIGHT_MAP = {0: "5 lb", 1: "10 lb"}

DOWN_ANGLE_THRESHOLD = 150
UP_ANGLE_THRESHOLD = 50
MAX_SHOULDER_MOVEMENT = 0.08
MAX_ELBOW_DRIFT = 0.15
MIN_ROM_ANGLE_CHANGE = 80

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def get_landmark(landmarks, idx, w, h):
    if landmarks is None:
        return None
    lm = landmarks.landmark[idx]
    if lm.visibility < 0.5:
        return None
    return (int(lm.x * w), int(lm.y * h))


def draw_text_bg(frame, text, pos, scale=0.7, color=(255, 255, 255), bg=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    x, y = pos
    cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5), bg, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, 2)


class FormAnalyzer:
    def __init__(self, is_right_arm=True):
        self.is_right_arm = is_right_arm
        self.reset()

    def reset(self):
        self.shoulder_baseline = None
        self.elbow_baseline_x = None
        self.hip_baseline = None
        self.min_angle = 180
        self.max_angle = 0
        self.errors = []
        self.pose_history = deque(maxlen=30)

    def update(self, landmarks, w, h):
        use_right = self.is_right_arm
        shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER if use_right else mp_pose.PoseLandmark.LEFT_SHOULDER
        elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW if use_right else mp_pose.PoseLandmark.LEFT_ELBOW
        wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST if use_right else mp_pose.PoseLandmark.LEFT_WRIST
        hip_idx = mp_pose.PoseLandmark.RIGHT_HIP if use_right else mp_pose.PoseLandmark.LEFT_HIP

        shoulder = get_landmark(landmarks, shoulder_idx, w, h)
        elbow = get_landmark(landmarks, elbow_idx, w, h)
        wrist = get_landmark(landmarks, wrist_idx, w, h)
        hip = get_landmark(landmarks, hip_idx, w, h)

        if not all([shoulder, elbow, wrist, hip]):
            return None, "Keypoints not visible"

        angle = calculate_angle(shoulder, elbow, wrist)
        self.min_angle = min(self.min_angle, angle)
        self.max_angle = max(self.max_angle, angle)

        self.pose_history.append({
            'shoulder': shoulder, 'elbow': elbow,
            'wrist': wrist, 'hip': hip, 'angle': angle
        })

        if self.shoulder_baseline is None:
            self.shoulder_baseline = shoulder[1]
            self.elbow_baseline_x = elbow[0]
            self.hip_baseline = hip[1]

        self.errors = []

        shoulder_move = abs(shoulder[1] - self.shoulder_baseline) / h
        if shoulder_move > MAX_SHOULDER_MOVEMENT:
            self.errors.append("Shoulder moving")

        elbow_drift = abs(elbow[0] - self.elbow_baseline_x) / w
        if elbow_drift > MAX_ELBOW_DRIFT:
            self.errors.append("Elbow drifting")

        return angle, self.errors if self.errors else "Good form"

    def check_rep_validity(self):
        rom = self.max_angle - self.min_angle
        if rom < MIN_ROM_ANGLE_CHANGE:
            return False, f"Incomplete ROM ({int(rom)}deg)"
        if self.errors:
            return False, "; ".join(self.errors)
        return True, "Valid rep"

    def get_snapshot(self):
        if self.pose_history:
            return list(self.pose_history)
        return None


class BicepCurlTracker:
    def __init__(self, reps_target=8, rest_interval=30, sets_target=3, dual_arm_mode=True):
        self.reps_target = reps_target
        self.rest_interval = rest_interval
        self.sets_target = sets_target
        self.dual_arm_mode = dual_arm_mode

        self.current_reps = 0
        self.current_sets = 0
        self.invalid_reps = 0

        self.left_state = "down"
        self.right_state = "down"
        self.left_rep_complete = False
        self.right_rep_complete = False

        self.rep_counted_this_frame = False

        self.weight_detected = "Unknown"

        self.running = False
        self.cap = None

        self.left_analyzer = FormAnalyzer(is_right_arm=False)
        self.right_analyzer = FormAnalyzer(is_right_arm=True)

        self.correct_form_snapshot = None
        self.incorrect_form_snapshot = None
        self.last_rep_valid = True

        self.apriltag_detector = None
        if APRILTAG_AVAILABLE:
            if APRILTAG_TYPE == "apriltag":
                self.apriltag_detector = apriltag.Detector()
            else:
                self.apriltag_detector = AprilDetector(families="tag36h11")
        # print(f"AprilTag Detector: {APRILTAG_TYPE if APRILTAG_AVAILABLE else 'Not Available'}")

    def detect_weight(self, gray):
        if not self.apriltag_detector:
            return
        try:
            if APRILTAG_TYPE == "apriltag":
                detections = self.apriltag_detector.detect(gray)
            else:
                detections = self.apriltag_detector.detect(gray)
            # print(f"Detections: {detections}")
            for d in detections:
                tag_id = d.tag_id if hasattr(d, 'tag_id') else d.tag_id
                # print(f"Detected Tag ID: {tag_id}")
                if tag_id in TAG_WEIGHT_MAP:
                    self.weight_detected = TAG_WEIGHT_MAP[tag_id]
                    break
        except:
            pass

    def draw_skeleton_overlay(self, frame, pose_data, color, alpha=0.6):
        if not pose_data:
            return
        overlay = frame.copy()
        for p in pose_data:
            pts = [p['shoulder'], p['elbow'], p['wrist']]
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i + 1], color, 4)
            for pt in pts:
                cv2.circle(overlay, pt, 8, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def show_correct_form_demo(self, pose):
        demo_start = time.time()
        demo_duration = 5

        # Use right arm for demo
        demo_analyzer = FormAnalyzer(is_right_arm=True)

        while self.running and (time.time() - demo_start) < demo_duration:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                demo_analyzer.update(results.pose_landmarks, w, h)
                self.correct_form_snapshot = demo_analyzer.get_snapshot()

            remaining = int(demo_duration - (time.time() - demo_start))
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
            draw_text_bg(frame, "CORRECT FORM DEMONSTRATION", (10, 30), 0.8, (0, 255, 0))
            draw_text_bg(frame, f"Hold proper curl position - {remaining}s", (10, 60), 0.6)
            draw_text_bg(frame, "Press 's' to skip", (w - 200, 30), 0.5)

            cv2.imshow('Bicep Curl Tracker', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('s'):
                break

    def show_rest_period(self, frame_shape):
        rest_start = time.time()
        h, w = frame_shape[:2]

        while self.running and (time.time() - rest_start) < self.rest_interval:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            remaining = int(self.rest_interval - (time.time() - rest_start))

            if self.correct_form_snapshot:
                self.draw_skeleton_overlay(frame, self.correct_form_snapshot, (0, 255, 0), 0.4)
            if self.incorrect_form_snapshot and not self.last_rep_valid:
                self.draw_skeleton_overlay(frame, self.incorrect_form_snapshot, (0, 0, 255), 0.4)

            cv2.rectangle(frame, (0, h // 2 - 60), (w, h // 2 + 60), (0, 0, 0), -1)
            draw_text_bg(frame, f"REST: {remaining}s", (w // 2 - 80, h // 2 - 20), 1.5, (0, 255, 255))
            draw_text_bg(frame, f"Set {self.current_sets}/{self.sets_target} complete", (w // 2 - 100, h // 2 + 30),
                         0.7)

            if self.correct_form_snapshot and self.incorrect_form_snapshot:
                draw_text_bg(frame, "GREEN=Correct  RED=Your form", (10, h - 30), 0.5)

            cv2.imshow('Bicep Curl Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

    def draw_stats_overlay(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (280, 130), (0, 0, 0), -1)
        draw_text_bg(frame, f"Weight: {self.weight_detected}", (10, 25), 0.6, (255, 255, 0))
        draw_text_bg(frame, f"Set: {self.current_sets + 1}/{self.sets_target}", (10, 50), 0.6)
        draw_text_bg(frame, f"Reps: {self.current_reps}/{self.reps_target}", (10, 75), 0.6)
        draw_text_bg(frame, f"Invalid: {self.invalid_reps}", (10, 100), 0.6, (0, 0, 255))

        mode_text = "Dual Arm Mode" if self.dual_arm_mode else "Single Arm Mode"
        draw_text_bg(frame, mode_text, (w - 150, 25), 0.5, (255, 255, 0))

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def process_arm(self, analyzer, state, landmarks, w, h, is_right):
        angle, feedback = analyzer.update(landmarks, w, h)
        new_state = state

        if angle is not None:
            # Ensure vertical movement by checking relative y-coordinates
            shoulder_y = landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_SHOULDER if is_right else mp_pose.PoseLandmark.LEFT_SHOULDER].y
            wrist_y = landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_WRIST if is_right else mp_pose.PoseLandmark.LEFT_WRIST].y

            # Get landmarks
            shoulder = landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_SHOULDER if is_right else mp_pose.PoseLandmark.LEFT_SHOULDER]
            wrist = landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_WRIST if is_right else mp_pose.PoseLandmark.LEFT_WRIST]

            # Horizontal movement: check x-axis displacement relative to shoulder
            x_diff = abs(wrist.x - shoulder.x)
            y_diff = abs(wrist.y - shoulder.y)  # optional, for vertical alignment check

            HORIZONTAL_THRESHOLD = 0.15  # adjust based on testing

            if x_diff > HORIZONTAL_THRESHOLD:
                feedback = "Horizontal movement detected"

            if isinstance(feedback, list) and feedback:
                form_text = "; ".join(feedback)
                form_color = (0, 0, 255)
            elif feedback == "Good form":
                form_text = "Good form"
                form_color = (0, 255, 0)
            else:
                form_text = str(feedback)
                form_color = (0, 255, 255)

            if state == "down" and angle < UP_ANGLE_THRESHOLD:
                new_state = "up"
            elif state == "up" and angle > DOWN_ANGLE_THRESHOLD:
                valid, reason = analyzer.check_rep_validity()

                if valid and x_diff < HORIZONTAL_THRESHOLD:
                    if self.dual_arm_mode:
                        if is_right:
                            self.right_rep_complete = True
                        else:
                            self.left_rep_complete = True

                        if self.right_rep_complete and self.left_rep_complete:
                            self.current_reps += 1
                            self.right_rep_complete = False
                            self.left_rep_complete = False
                    else:
                        # self.current_reps += 1
                        if not self.rep_counted_this_frame:
                            self.current_reps += 1
                            self.rep_counted_this_frame = True

                    self.last_rep_valid = True
                else:
                    self.invalid_reps += 1
                    self.last_rep_valid = False
                    self.incorrect_form_snapshot = analyzer.get_snapshot()

                analyzer.reset()
                new_state = "down"

        return new_state, angle, feedback

    def _run_loop(self):
        pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

        self.show_correct_form_demo(pose)

        while self.running:
            ret, frame1 = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame1, 1)
            h, w = frame.shape[:2]

            # print( f"detecting weight" )
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            self.detect_weight(gray)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

                # Process Right Arm
                self.right_state, r_angle, r_feedback = self.process_arm(
                    self.right_analyzer, self.right_state, results.pose_landmarks, w, h, True
                )

                # Process Left Arm
                self.left_state, l_angle, l_feedback = self.process_arm(
                    self.left_analyzer, self.left_state, results.pose_landmarks, w, h, False
                )

                # Display Angles
                if r_angle is not None:
                    draw_text_bg(frame, f"R: {int(r_angle)}", (w - 150, 60), 0.6)
                if l_angle is not None:
                    draw_text_bg(frame, f"L: {int(l_angle)}", (w - 150, 90), 0.6)

                # Display Feedback (Prioritize error, or show both?)
                # Simple approach: Show combined or separate. Let's show separate at bottom.
                fb_y = h - 30
                if r_feedback != "Good form" and r_feedback != "Keypoints not visible":
                    draw_text_bg(frame, f"R: {r_feedback}", (10, fb_y), 0.6, (0, 0, 255))
                    fb_y -= 30
                if l_feedback != "Good form" and l_feedback != "Keypoints not visible":
                    draw_text_bg(frame, f"L: {l_feedback}", (10, fb_y), 0.6, (0, 0, 255))

                if self.current_reps >= self.reps_target:
                    self.current_sets += 1
                    self.current_reps = 0

                    if self.current_sets >= self.sets_target:
                        self.running = False
                        cv2.rectangle(frame, (0, h // 2 - 40), (w, h // 2 + 40), (0, 0, 0), -1)
                        draw_text_bg(frame, "WORKOUT COMPLETE!", (w // 2 - 150, h // 2 + 10), 1.2, (0, 255, 0))
                        cv2.imshow('Bicep Curl Tracker', frame)
                        cv2.waitKey(3000)
                        break
                    else:
                        self.show_rest_period(frame.shape)

            self.draw_stats_overlay(frame)
            draw_text_bg(frame, "Q=Quit", (w - 100, h - 20), 0.5)

            cv2.imshow('Bicep Curl Tracker', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break

            self.rep_counted_this_frame = False

        pose.close()
        self.stop()


class AppUI:
    def __init__(self, root):
        self.root = root
        root.title('Bicep Curl Tracker')
        root.geometry('350x300')
        root.resizable(False, False)

        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 11))
        style.configure('TButton', font=('Arial', 11))

        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill='both', expand=True)

        ttk.Label(main_frame, text='Bicep Curl Tracker', font=('Arial', 16, 'bold')).grid(row=0, column=0, columnspan=2,
                                                                                          pady=(0, 15))

        ttk.Label(main_frame, text='Repetitions per set:').grid(row=1, column=0, sticky='w', pady=5)
        self.reps_var = tk.IntVar(value=8)
        ttk.Spinbox(main_frame, from_=1, to=50, textvariable=self.reps_var, width=10).grid(row=1, column=1, pady=5)

        ttk.Label(main_frame, text='Rest interval (sec):').grid(row=2, column=0, sticky='w', pady=5)
        self.rest_var = tk.IntVar(value=30)
        ttk.Spinbox(main_frame, from_=5, to=120, textvariable=self.rest_var, width=10).grid(row=2, column=1, pady=5)

        ttk.Label(main_frame, text='Number of sets:').grid(row=3, column=0, sticky='w', pady=5)
        self.sets_var = tk.IntVar(value=3)
        ttk.Spinbox(main_frame, from_=1, to=10, textvariable=self.sets_var, width=10).grid(row=3, column=1, pady=5)
        self.dual_arm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, variable=self.dual_arm_var).grid(row=4, column=1, pady=5)
        mode_text = "Dual Arm Mode" if self.dual_arm_var.get() else "Single Arm Mode"
        ttk.Label(main_frame, text=mode_text).grid(row=4, column=0, sticky='w', pady=5)

        ttk.Button(main_frame, text='Start Workout', command=self.start_workout).grid(row=5, column=0, columnspan=2,
                                                                                      pady=20)

        ttk.Label(main_frame, text='Q=Quit, S=Skip Demo', font=('Arial', 9)).grid(row=6, column=0, columnspan=2)

        self.tracker = None

    def start_workout(self):
        reps = self.reps_var.get()
        rest = self.rest_var.get()
        sets = self.sets_var.get()
        dual_arm_mode = self.dual_arm_var.get()

        if reps <= 0 or rest < 0 or sets <= 0:
            messagebox.showerror('Invalid Input', 'Please enter valid positive numbers')
            return

        self.tracker = BicepCurlTracker(
            reps_target=reps,
            rest_interval=rest,
            sets_target=sets,
            dual_arm_mode=dual_arm_mode
        )

        try:
            self.tracker.start()
        except Exception as e:
            messagebox.showerror('Error', str(e))


if __name__ == '__main__':
    root = tk.Tk()
    app = AppUI(root)
    root.mainloop()
