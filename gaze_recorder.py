"""
Gaze + blink recorder based on EyeGestures v3.

Usage:
    python gaze_recorder.py [participant_id]

Flow:
    1. Intro screen    -> press SPACE to start calibration
    2. Calibration     -> look at the centre of each dot until it moves (36 points)
    3. Accuracy check  -> look at 9 green dots; the error is measured and corrected
                          (SPACE = record, R = redo calibration, V = redo check)
    4. Recording       -> runs until you press ESC / Ctrl+Q, or the data limit is reached

Output (in the "recordings" folder next to this script):
    <id>_<date>_<time>_gaze.csv    timestamp, t_s, x, y (corrected), raw_x, raw_y, on_screen,
                                   fixation, blink, face_detected, ear
    <id>_<date>_<time>_blinks.csv  start_timestamp, start_t_s, end_t_s, duration_ms
    <id>_<date>_<time>_meta.json   screen size, accuracy check results, correction

Blinks are detected with the eye aspect ratio (EAR = eye height / eye width, from the
MediaPipe face landmarks). A blink is when EAR drops below BLINK_RATIO x the person's
normal open-eye EAR, which is learned continuously over the last few seconds.
"""

import collections
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

if __name__ == "__main__" and os.environ.get("EYEGESTURES_RECORDER_MANAGED") != "1":
    _root = os.path.dirname(os.path.abspath(__file__))
    _launcher = os.path.join(_root, "run_gaze_recorder.py")
    raise SystemExit(subprocess.run([sys.executable, _launcher, *sys.argv[1:]], cwd=_root).returncode)

import cv2
import numpy as np
import pygame

# The repo's own eyeGestures/ source folder sits next to this script and would shadow
# the installed pip package (which is the one that has EyeGestures_v3).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]

from eyeGestures import EyeGestures_v3
from eyeGestures.utils import VideoCapture

# ---------------- Settings ----------------
SAMPLE_INTERVAL_S = 1 / 30            # how often a gaze sample is saved (seconds)
MAX_BYTES = 300 * 1024 * 1024         # stop recording when both files together reach this size
MIN_BLINK_S = 0.0                     # ignore blinks shorter than this (filters flicker), e.g. 0.05
MAX_BLINK_S = 1.0                     # eye "closures" longer than this are not saved as blinks
BLINK_RATIO = 0.70                    # blink when EAR < this x open-eye EAR. Higher = more sensitive
BLINK_BASELINE_S = 3.0                # open-eye EAR is learned from the last this many seconds
SHOW_EAR = True                       # show live EAR value and threshold on screen (for tuning)
MIRROR = True                         # flip camera image horizontally (like the examples)
SHOW_GAZE_DOT = False                 # draw the gaze point on screen while recording
CAL_GRID = 6                          # calibration grid is CAL_GRID x CAL_GRID points
CAL_SETTLE_S = 0.6                    # wait this long after a dot moves before collecting samples
VAL_MARGIN = 0.05                     # accuracy-check dots sit this far (fraction) from screen edges
VAL_POINT_S = 2.0                     # how long each accuracy-check dot is shown
VAL_SETTLE_S = 0.8                    # ignore the first part of each check dot (eyes still moving)
APPLY_CORRECTION = True               # correct gaze with the accuracy-check results
TARGET_RADIUS = 22
UI_SCALE = 1.0                        # extra size factor for text and dots (e.g. 1.3 = 30% bigger)
CAMERA_INDEX = 0
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
# ------------------------------------------

CONTEXT = "recorder"
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (100, 0, 255)
RED = (255, 0, 100)
GREEN = (0, 220, 100)
GREY = (120, 120, 120)


def get_participant_id():
    pid = sys.argv[1] if len(sys.argv) > 1 else input("Participant ID: ")
    pid = re.sub(r"[^A-Za-z0-9_-]", "_", pid.strip())
    return pid or "anonymous"


# MediaPipe face-mesh indices: corner, top, top, corner, bottom, bottom
_EYES = ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380))


def eye_aspect_ratio(landmarks):
    """Average EAR of both eyes; None if landmarks are unavailable."""
    if landmarks is None or len(landmarks) < 388:
        return None
    ears = []
    for p1, p2, p3, p4, p5, p6 in _EYES:
        lm = landmarks
        width = np.linalg.norm(lm[p1] - lm[p4])
        if width < 1e-6:
            return None
        ears.append((np.linalg.norm(lm[p2] - lm[p6]) + np.linalg.norm(lm[p3] - lm[p5])) / (2 * width))
    return float(np.mean(ears))


class BlinkDetector:
    """Adaptive EAR threshold: blink = EAR below BLINK_RATIO x recent open-eye EAR.

    The open-eye EAR depends on where you look (lower when looking down) and on head pose,
    so the baseline is the 80th percentile of the last BASELINE_S seconds of *all* samples.
    Short blinks barely move that percentile, but a lasting change (e.g. reading lower on
    the screen) is adopted within a couple of seconds, so the detector can't get stuck closed.
    """

    def __init__(self, ratio, window_s):
        self.ratio = ratio
        self.window_s = window_s
        self.history = collections.deque()  # (time, ear)
        self.closed = False

    def threshold(self):
        if len(self.history) < 15:
            return None
        return float(np.percentile([e for _, e in self.history], 80)) * self.ratio

    def update(self, ear, now=None):
        if ear is None:
            return self.closed
        now = time.time() if now is None else now
        self.history.append((now, ear))
        while self.history and now - self.history[0][0] > self.window_s:
            self.history.popleft()
        thr = self.threshold()
        if thr is None:
            return False
        # hysteresis: reopen only when clearly above threshold, avoids flicker
        if self.closed:
            self.closed = ear < thr * 1.1
        else:
            self.closed = ear < thr
        return self.closed


def iso(t):
    return dt.datetime.fromtimestamp(t).isoformat(timespec="milliseconds")


def draw_text_lines(screen, font, lines, color=WHITE):
    w, h = screen.get_size()
    total = len(lines) * font.get_linesize()
    y = (h - total) // 2
    for line in lines:
        surf = font.render(line, True, color)
        screen.blit(surf, surf.get_rect(midtop=(w // 2, y)))
        y += font.get_linesize()


def fit_correction(raw, true):
    """Affine map raw gaze -> true screen position (least squares). Returns 3x2 matrix."""
    A = np.column_stack([raw, np.ones(len(raw))])
    M, *_ = np.linalg.lstsq(A, true, rcond=None)
    return M


def apply_correction(M, x, y):
    if M is None:
        return x, y
    cx, cy = np.array([x, y, 1.0]) @ M
    return float(cx), float(cy)


def new_tracker(width, height):
    gestures = EyeGestures_v3()
    grid = np.linspace(0.0, 1.0, CAL_GRID)
    xx, yy = np.meshgrid(grid, grid)
    cal_map = np.column_stack([xx.ravel(), yy.ravel()])
    np.random.shuffle(cal_map)
    gestures.uploadCalibrationMap(cal_map, context=CONTEXT)
    gestures.setFixation(1.0)
    return gestures, len(cal_map)


def draw_target(screen, point, color, label=None, font=None):
    pygame.draw.circle(screen, color, point, TARGET_RADIUS, 3)
    pygame.draw.circle(screen, WHITE, point, 5)
    if label and font:
        surf = font.render(label, True, GREY)
        screen.blit(surf, surf.get_rect(midtop=(point[0], point[1] + TARGET_RADIUS + 8)))


def clamp_point(p, width, height, margin):
    return (int(min(max(p[0], margin), width - margin)), int(min(max(p[1], margin), height - margin)))


def main():
    pid = get_participant_id()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.join(OUTPUT_DIR, f"{pid}_{dt.datetime.now():%Y%m%d_%H%M%S}")
    gaze_path, blink_path, meta_path = base + "_gaze.csv", base + "_blinks.csv", base + "_meta.json"

    global TARGET_RADIUS
    ui = 1.0
    if sys.platform == "win32":
        # use real pixels, so Windows display scaling doesn't change the screen or gaze coordinates
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        try:
            # ...but keep text and dots as big as Windows scaling would make them
            ui = ctypes.windll.user32.GetDpiForSystem() / 96
        except Exception:
            pass
    ui *= UI_SCALE
    TARGET_RADIUS = round(TARGET_RADIUS * ui)
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    pygame.init()
    width, height = pygame.display.get_desktop_sizes()[0]
    # borderless window covering the screen instead of exclusive fullscreen (no display mode change)
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
    pygame.display.set_caption("Gaze recorder")
    font = pygame.font.Font(None, round(48 * ui))
    small = pygame.font.Font(None, round(28 * ui))
    clock = pygame.time.Clock()

    screen.fill(BLACK)
    draw_text_lines(screen, font, ["Loading camera and face tracker..."], GREY)
    pygame.display.flip()
    pygame.event.pump()

    gestures, n_points = new_tracker(width, height)
    cap = VideoCapture(CAMERA_INDEX)
    blinker = BlinkDetector(BLINK_RATIO, BLINK_BASELINE_S)

    val_targets = [(fx * width, fy * height)
                   for fy in (VAL_MARGIN, 0.5, 1 - VAL_MARGIN)
                   for fx in (VAL_MARGIN, 0.5, 1 - VAL_MARGIN)]

    state = "intro"
    cal_count = 0
    prev_cal = None
    cal_changed_at = 0.0
    val_index = 0
    val_started = 0.0
    val_samples = []
    val_medians = []
    correction = None
    val_result = {}
    gaze_file = blink_file = gaze_writer = blink_writer = None
    t0 = next_sample = next_flush = 0.0
    blink_start = None
    n_samples = n_blinks = 0
    stop_reason = "stopped by user"

    def write_blink(start, end):
        nonlocal n_blinks
        if MIN_BLINK_S <= end - start <= MAX_BLINK_S:
            blink_writer.writerow([iso(start), f"{start - t0:.3f}", f"{end - t0:.3f}",
                                   round((end - start) * 1000)])
            n_blinks += 1

    def start_calibration():
        nonlocal gestures, n_points, state, cal_count, prev_cal, correction
        gestures, n_points = new_tracker(width, height)
        state, cal_count, prev_cal, correction = "calibrating", 0, None, None

    def start_validation():
        nonlocal state, val_index, val_started, val_samples, val_medians
        state, val_index, val_started, val_samples, val_medians = "validating", 0, time.time(), [], []

    def finish_validation():
        nonlocal state, correction, val_result
        pairs = [(m, t) for m, t in zip(val_medians, val_targets) if m is not None]
        state = "val_result"
        val_result = {"valid_points": len(pairs)}
        if len(pairs) < 4:
            correction = None
            return
        raw = np.array([p[0] for p in pairs])
        true = np.array([p[1] for p in pairs])
        before = np.sqrt(np.mean(np.sum((raw - true) ** 2, axis=1)))
        if APPLY_CORRECTION:
            correction = fit_correction(raw, true)
            fixed = np.column_stack([raw, np.ones(len(raw))]) @ correction
            after = np.sqrt(np.mean(np.sum((fixed - true) ** 2, axis=1)))
        else:
            after = before
        val_result.update(error_before_px=round(float(before)), error_after_px=round(float(after)),
                          raw_medians=raw.round(1).tolist(), targets=true.round(1).tolist())

    def start_recording():
        nonlocal state, gaze_file, blink_file, gaze_writer, blink_writer, t0, next_sample, next_flush
        state = "recording"
        gaze_file = open(gaze_path, "w", newline="", encoding="utf-8")
        blink_file = open(blink_path, "w", newline="", encoding="utf-8")
        gaze_writer = csv.writer(gaze_file)
        blink_writer = csv.writer(blink_file)
        gaze_writer.writerow(["timestamp", "t_s", "x", "y", "raw_x", "raw_y", "on_screen",
                              "fixation", "blink", "face_detected", "ear"])
        blink_writer.writerow(["start_timestamp", "start_t_s", "end_t_s", "duration_ms"])
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "participant": pid,
                "screen_width": width,
                "screen_height": height,
                "started": iso(time.time()),
                "calibration_points": n_points,
                "validation": val_result,
                "correction_matrix": None if correction is None else correction.tolist(),
                "blink_ratio": BLINK_RATIO,
            }, f, indent=2)
        t0 = next_sample = time.time()
        next_flush = t0 + 1.0

    running = True
    try:
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE or (e.key == pygame.K_q and e.mod & pygame.KMOD_CTRL):
                        running = False
                    elif e.key == pygame.K_SPACE and state == "intro":
                        start_calibration()
                    elif e.key == pygame.K_SPACE and state == "val_result":
                        start_recording()
                    elif e.key == pygame.K_r and state == "val_result":
                        start_calibration()
                    elif e.key == pygame.K_v and state == "val_result":
                        start_validation()
            if not running:
                break

            screen.fill(BLACK)

            if state in ("intro", "val_result"):
                if state == "intro":
                    lines = [
                        f"Participant: {pid}",
                        "",
                        "Calibration: look at the CENTRE of each dot until it moves.",
                        "Keep your head still and don't blink much while a dot is shown.",
                        "",
                        "Press SPACE to start calibration, ESC to quit.",
                    ]
                elif "error_after_px" in val_result:
                    lines = [
                        f"Accuracy check: average error {val_result['error_after_px']} px "
                        f"(uncorrected {val_result['error_before_px']} px)",
                        "",
                        "SPACE = start recording    R = redo calibration    V = redo check    ESC = quit",
                    ]
                else:
                    lines = [
                        "Accuracy check failed (face not detected at enough points).",
                        "",
                        "R = redo calibration    V = redo check    SPACE = record anyway    ESC = quit",
                    ]
                draw_text_lines(screen, font, lines)
                if state == "val_result" and "raw_medians" in val_result:
                    for raw, tgt in zip(val_result["raw_medians"], val_result["targets"]):
                        shown = apply_correction(correction, *raw)
                        pygame.draw.circle(screen, GREY, (int(tgt[0]), int(tgt[1])), 6)
                        pygame.draw.line(screen, GREY, (int(tgt[0]), int(tgt[1])),
                                         clamp_point(shown, width, height, 0), 1)
                        pygame.draw.circle(screen, RED, clamp_point(shown, width, height, 6), 6)
                pygame.display.flip()
                clock.tick(30)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if MIRROR:
                frame = np.flip(frame, axis=1)

            now = time.time()
            calibrate = False
            if state == "calibrating":
                target = gestures.clb[CONTEXT].getCurrentPoint(width, height)
                point = (int(target[0]), int(target[1]))
                if point != prev_cal:
                    cal_count += 1
                    prev_cal = point
                    cal_changed_at = now
                if cal_count > n_points:
                    start_validation()
                else:
                    # give the eyes time to reach the new dot before collecting samples
                    calibrate = now - cal_changed_at >= CAL_SETTLE_S

            try:
                event, calibration = gestures.step(frame, calibrate, width, height, context=CONTEXT)
                ear = eye_aspect_ratio(gestures.face.getLandmarks())
            except Exception:
                event = calibration = ear = None
            face = event is not None
            blink_closed = blinker.update(ear, now) if face else False

            if state == "calibrating":
                shown = clamp_point(prev_cal, width, height, TARGET_RADIUS + 4)
                color = BLUE if calibrate else GREY
                draw_target(screen, shown, color, f"{cal_count}/{n_points}", small)
                if not face:
                    draw_text_lines(screen, font, ["Face not detected"], GREY)

            elif state == "validating":
                tgt = val_targets[val_index]
                elapsed = now - val_started
                if face and not blink_closed and elapsed >= VAL_SETTLE_S:
                    val_samples.append((float(event.point[0]), float(event.point[1])))
                if elapsed >= VAL_POINT_S:
                    val_medians.append(tuple(np.median(val_samples, axis=0)) if len(val_samples) >= 5 else None)
                    val_samples = []
                    val_index += 1
                    val_started = now
                    if val_index >= len(val_targets):
                        finish_validation()
                        pygame.display.flip()
                        continue
                    tgt = val_targets[val_index]
                draw_target(screen, (int(tgt[0]), int(tgt[1])), GREEN,
                            f"check {val_index + 1}/{len(val_targets)}", small)
                if not face:
                    draw_text_lines(screen, font, ["Face not detected"], GREY)

            elif state == "recording":
                t = now
                blink = bool(face and blink_closed)

                if blink and blink_start is None:
                    blink_start = t
                elif not blink and blink_start is not None:
                    write_blink(blink_start, t)
                    blink_start = None

                gx = gy = None
                if face:
                    rx, ry = float(event.point[0]), float(event.point[1])
                    gx, gy = apply_correction(correction, rx, ry)

                if t >= next_sample:
                    next_sample = max(next_sample + SAMPLE_INTERVAL_S, t)
                    if face:
                        on = int(0 <= gx < width and 0 <= gy < height)
                        row = [int(gx), int(gy), int(rx), int(ry), on, round(float(event.fixation), 3)]
                    else:
                        row = ["", "", "", "", "", ""]
                    gaze_writer.writerow([iso(t), f"{t - t0:.3f}"] + row +
                                         [int(blink), int(face), "" if ear is None else f"{ear:.3f}"])
                    n_samples += 1

                if t >= next_flush:
                    next_flush = t + 1.0
                    gaze_file.flush()
                    blink_file.flush()
                    if os.path.getsize(gaze_path) + os.path.getsize(blink_path) >= MAX_BYTES:
                        stop_reason = "data limit reached"
                        running = False

                if SHOW_GAZE_DOT and gx is not None:
                    pygame.draw.circle(screen, RED, clamp_point((gx, gy), width, height, 15), 15)
                elapsed = int(t - t0)
                status = small.render(
                    f"Recording {elapsed // 60:02d}:{elapsed % 60:02d}   blinks: {n_blinks}   ESC to stop",
                    True, GREY)
                screen.blit(status, (10, 10))
                if SHOW_EAR:
                    thr = blinker.threshold()
                    debug = small.render(
                        f"EAR {ear if ear is not None else 0:.3f}   threshold "
                        f"{thr if thr is not None else 0:.3f}   {clock.get_fps():.0f} fps"
                        + ("   BLINK" if blink else ""), True, RED if blink else GREY)
                    screen.blit(debug, (10, 20 + small.get_linesize()))
                if not face:
                    draw_text_lines(screen, font, ["Face not detected"], GREY)

            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        pass
    finally:
        if blink_writer is not None and blink_start is not None:
            write_blink(blink_start, time.time())
        for f in (gaze_file, blink_file):
            if f is not None:
                f.close()
        cap.close()
        pygame.quit()

    if gaze_file is None:
        print("Stopped before recording started; nothing saved.")
    else:
        print(f"Recording ended ({stop_reason}).")
        if "error_after_px" in val_result:
            print(f"  accuracy check: ~{val_result['error_after_px']} px average error")
        print(f"  {n_samples} gaze samples -> {gaze_path}")
        print(f"  {n_blinks} blinks        -> {blink_path}")
        print(f"  settings      -> {meta_path}")


if __name__ == "__main__":
    main()
