"""
╔══════════════════════════════════════════════════════════╗
║          YOLOv26 Smart Traffic Counter  v2.0             ║
║     Real-time vehicle detection + road capacity alert    ║
╚══════════════════════════════════════════════════════════╝
"""

import cv2
import time
import sys
import os
from ultralytics import YOLO
from collections import defaultdict, deque

# ─── Colour palette (BGR) ────────────────────────────────
C_GREEN   = (57,  255, 20)    # neon green
C_YELLOW  = (0,   220, 255)   # amber/yellow
C_RED     = (40,  40,  255)   # vivid red
C_CYAN    = (255, 220, 0)     # cyan-blue
C_WHITE   = (240, 240, 240)
C_BLACK   = (0,   0,   0)
C_PANEL   = (15,  15,  15)    # near-black for glass panel

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ─── Helpers ─────────────────────────────────────────────

def draw_filled_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.55, radius=12):
    """Draw a semi-transparent rounded-corner rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_border_rect(img, x1, y1, x2, y2, color, thickness=2):
    """Thin glowing border."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def put_text_shadow(img, text, pos, font, scale, color, thickness=2):
    """Text with subtle drop-shadow for readability."""
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     font, scale, color,   thickness,   cv2.LINE_AA)


def get_crowd_status(count_per_min: int, limit: int):
    """Return (label, color) based on traffic vs road capacity."""
    diff = count_per_min - limit
    if diff <= 0:
        return "✅  NO CROWD", C_GREEN
    elif diff <= 10:
        return "⚠   SLIGHTLY CROWDED", C_YELLOW
    else:
        return "🚨  MORE CROWDED", C_RED


def ask_capacity() -> int:
    """Prompt user for road capacity in the terminal."""
    print("\n" + "═"*54)
    print("  YOLOv26 Smart Traffic Counter — Road Capacity Setup")
    print("═"*54)
    while True:
        try:
            cap = int(input("  Enter road capacity (max vehicles per minute): ").strip())
            if cap <= 0:
                raise ValueError
            print(f"  ✔ Limit set to {cap} vehicles/min\n")
            return cap
        except ValueError:
            print("  ✗ Please enter a valid positive integer.")


# ─── Main ────────────────────────────────────────────────

def main():
    # ── Paths ──────────────────────────────────────────
    ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VIDEO_PATH   = os.path.join(ROOT, "assets", "video.mp4")
    MODEL_PATH   = os.path.join(ROOT, "models", "yolo26n.pt")
    OUTPUT_PATH  = os.path.join(ROOT, "assets", "output.mp4")

    # ── User input: road capacity ──────────────────────
    road_limit = ask_capacity()

    # ── Load model & video ─────────────────────────────
    print("  Loading YOLO26 model …")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (W, H))

    # ── Counting state ─────────────────────────────────
    line_y      = int(H * 0.50)        # counting line at 50 % height
    track_hist  = defaultdict(list)
    crossed_ids = set()

    # Sliding window: timestamps of each crossing (for per-minute rate)
    crossing_times: deque = deque()

    frame_idx   = 0
    fps_display = 0.0
    t_fps       = time.time()

    print("  Press  Q  to quit.\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # ── FPS calc every 30 frames ───────────────────
        if frame_idx % 30 == 0:
            fps_display = 30.0 / (time.time() - t_fps + 1e-9)
            t_fps = time.time()

        # ── YOLO tracking ─────────────────────────────
        results = model.track(frame, persist=True, show=False, verbose=False)

        now = time.time()

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, track_ids):
                x, y, bw, bh = box

                # Track history
                hist = track_hist[tid]
                hist.append((float(x), float(y)))
                if len(hist) > 30:
                    hist.pop(0)

                # Crossing detection
                if len(hist) > 2 and tid not in crossed_ids:
                    py, cy = hist[-2][1], hist[-1][1]
                    if (py < line_y <= cy) or (py > line_y >= cy):
                        crossed_ids.add(tid)
                        crossing_times.append(now)

                # Draw bounding box  (thin neon cyan)
                x1 = int(x - bw / 2); y1 = int(y - bh / 2)
                x2 = int(x + bw / 2); y2 = int(y + bh / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), C_CYAN, 2)
                put_text_shadow(frame, f"#{tid}", (x1, y1 - 8), FONT, 0.42, C_CYAN, 1)

        # ── Prune old crossings (> 60 s) ──────────────
        while crossing_times and now - crossing_times[0] > 60:
            crossing_times.popleft()

        per_min  = len(crossing_times)          # vehicles in last 60 s
        total    = len(crossed_ids)
        status_label, status_color = get_crowd_status(per_min, road_limit)

        # ── Draw counting line ─────────────────────────
        # Animated dash pattern (shifts every ~10 frames)
        dash_offset = (frame_idx // 10) % 20
        for sx in range(-dash_offset, W, 20):
            ex = min(sx + 10, W)
            if sx < 0: sx = 0
            cv2.line(frame, (sx, line_y), (ex, line_y), C_GREEN, 2)

        put_text_shadow(frame, "COUNTING LINE", (10, line_y - 10), FONT, 0.5, C_GREEN, 1)

        # ── HUD Panel (top-left) ───────────────────────
        panel_x1, panel_y1 = 10,  10
        panel_x2, panel_y2 = 360, 210

        draw_filled_rounded_rect(frame, panel_x1, panel_y1, panel_x2, panel_y2,
                                 C_PANEL, alpha=0.65)
        draw_border_rect(frame, panel_x1, panel_y1, panel_x2, panel_y2, C_CYAN, 2)

        # Title bar accent
        cv2.rectangle(frame, (panel_x1+2, panel_y1+2), (panel_x2-2, panel_y1+34), C_CYAN, -1)
        put_text_shadow(frame, "  YOLO26 TRAFFIC COUNTER",
                        (panel_x1+6, panel_y1+26), FONT_BOLD, 0.55, C_BLACK, 1)

        # Stats
        rows = [
            (f"TOTAL CROSSED  : {total}",   C_WHITE,  56),
            (f"LAST 60 s      : {per_min} veh/min", C_YELLOW, 86),
            (f"ROAD LIMIT     : {road_limit} veh/min",  C_WHITE,  116),
            (f"FPS            : {fps_display:.1f}",     C_GREEN,  146),
        ]
        for text, color, dy in rows:
            put_text_shadow(frame, text, (panel_x1+12, panel_y1+dy),
                            FONT, 0.52, color, 1)

        # ── Status banner (bottom strip) ──────────────
        bx1, by1 = 10, H - 70
        bx2, by2 = W - 10, H - 10
        draw_filled_rounded_rect(frame, bx1, by1, bx2, by2, C_PANEL, 0.72)
        draw_border_rect(frame, bx1, by1, bx2, by2, status_color, 3)

        # Centre the status text
        (tw, th), _ = cv2.getTextSize(status_label, FONT_BOLD, 0.85, 2)
        tx = (W - tw) // 2
        ty = by1 + (by2 - by1 + th) // 2
        put_text_shadow(frame, status_label, (tx, ty), FONT_BOLD, 0.85, status_color, 2)

        # diff annotation
        diff_txt = f"({per_min} vs {road_limit} limit)"
        (dw, _), _ = cv2.getTextSize(diff_txt, FONT, 0.4, 1)
        cv2.putText(frame, diff_txt, (W - dw - 20, by1 + 22),
                    FONT, 0.40, C_WHITE, 1, cv2.LINE_AA)

        # ── Write & display ────────────────────────────
        writer.write(frame)
        cv2.imshow("YOLOv26 Smart Traffic Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n  ✔ Done. Output saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
