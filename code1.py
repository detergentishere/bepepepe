#!/usr/bin/env python3
"""
mnist_air_live_ensemble_visual.py

3-CNN ensemble for MNIST air drawing.
Always drawing; left/right hand choice.
After quitting the webcam loop, shows test accuracy visualizations.
"""
import os
import time
import argparse
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "mediapipe is required. Install with: pip install mediapipe==0.10.8"
    ) from e

MODEL_DIR = "saved_models"
NUM_MODELS = 3
MODEL_NAME_TEMPLATE = "cnn_mnist_{}.keras"

FRAME_W = 640
FRAME_H = 480
CANVAS_W = FRAME_W
CANVAS_H = FRAME_H
BRUSH_RADIUS = 8
MIN_STROKE_PIXELS = 120
PRED_LOCK_SECONDS = 4.0
RESET_SECONDS = 2.0
SMOOTH_N = 6

# ----------------- Utilities -----------------
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_canvas_to_mnist(canvas):
    ys, xs = np.where(canvas > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    margin = 10
    x1 = max(0, x1 - margin); x2 = min(canvas.shape[1], x2 + margin)
    y1 = max(0, y1 - margin); y2 = min(canvas.shape[0], y2 + margin)
    roi = canvas[y1:y2, x1:x2]
    h, w = roi.shape
    if h == 0 or w == 0:
        return None
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))
    else:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas28 = np.zeros((28,28), dtype='uint8')
    xoff = (28 - new_w) // 2
    yoff = (28 - new_h) // 2
    canvas28[yoff:yoff+new_h, xoff:xoff+new_w] = roi_resized
    canvas28 = canvas28.astype('float32') / 255.0
    return canvas28.reshape(28,28,1)

def is_index_finger_up(landmarks):
    try:
        if landmarks.landmark[8].y < landmarks.landmark[6].y:
            folded = True
            for tip,pip in [(12,10),(16,14),(20,18)]:
                if landmarks.landmark[tip].y < landmarks.landmark[pip].y - 0.02:
                    folded = False
            return folded
    except Exception:
        return False
    return False

def is_thumbs_up(landmarks):
    try:
        if landmarks.landmark[4].y < landmarks.landmark[3].y - 0.01:
            folded = True
            for tip,pip in [(8,6),(12,10),(16,14),(20,18)]:
                if landmarks.landmark[tip].y < landmarks.landmark[pip].y - 0.02:
                    folded = False
            return folded
    except Exception:
        return False
    return False

# ----------------- Training / Loading -----------------
def train_and_save_ensemble(num_models=3, force_retrain=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    models_list = []

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype('float32') / 255.0)[...,None]
    x_test  = (x_test.astype('float32')  / 255.0)[...,None]
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test, 10)

    for i in range(num_models):
        model_path = os.path.join(MODEL_DIR, MODEL_NAME_TEMPLATE.format(i+1))
        if os.path.exists(model_path) and not force_retrain:
            print(f"Loading existing model {i+1}")
            models_list.append(tf.keras.models.load_model(model_path))
        else:
            print(f"Training model {i+1}...")
            model = build_cnn()
            ckpt = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy')
            es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            model.fit(x_train, y_train_cat, validation_split=0.1,
                      epochs=12, batch_size=128, callbacks=[ckpt, es], verbose=2)
            models_list.append(tf.keras.models.load_model(model_path))
        loss, acc = models_list[-1].evaluate(x_test, y_test_cat, verbose=0)
        print(f"Model {i+1} test accuracy: {acc:.4f}")
    return models_list

# ----------------- Live webcam loop -----------------
def run_live_loop(models_list, writing_hand_choice):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    if not cap.isOpened():
        for idx in (1,2,3):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == 'nt' else idx)
            if cap.isOpened():
                break
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    pts = deque(maxlen=1024)
    smooth_buf = deque(maxlen=SMOOTH_N)

    last_prediction = None
    last_conf = 0.0
    pred_locked = False
    prediction_time = 0.0
    reset_until = 0.0

    print("Webcam started. Draw with your index finger in the air.")
    print("Show thumbs-up to predict. Press 'c' to clear, 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            drawing_seen = False
            thumbs_up = False
            now = time.time()

            if now < reset_until:
                remaining = reset_until - now
                cv2.putText(frame, f"Resetting... {remaining:.1f}s", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,200), 2)
                if last_prediction is not None:
                    cv2.putText(frame, f"Last: {last_prediction} ({last_conf:.2f})", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
                preview = cv2.resize(canvas, (160,120))
                frame[10:10+120, W-10-160:W-10] = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                cv2.imshow("MNIST Air Draw", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                if key == ord('c'): canvas[:] = 0; pts.clear(); smooth_buf.clear(); last_prediction=None
                continue

            # hand tracking
            if res.multi_hand_landmarks and res.multi_handedness:
                for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    hand_label = handedness.classification[0].label
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if hand_label == writing_hand_choice and is_index_finger_up(hand_landmarks):
                        drawing_seen = True
                        ix = int(hand_landmarks.landmark[8].x * W)
                        iy = int(hand_landmarks.landmark[8].y * H)
                        smooth_buf.append((ix, iy))
                        sx = int(round(np.mean([p[0] for p in smooth_buf])))
                        sy = int(round(np.mean([p[1] for p in smooth_buf])))
                        pts.appendleft((sx, sy))
                        cv2.circle(frame, (sx, sy), 6, (0,255,0), -1)
                    elif hand_label == writing_hand_choice:
                        smooth_buf.clear()

                    if is_thumbs_up(hand_landmarks):
                        thumbs_up = True
                        tx = int(hand_landmarks.landmark[4].x * W)
                        ty = int(hand_landmarks.landmark[4].y * H)
                        cv2.putText(frame, "THUMBS UP", (tx+8, ty-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # draw strokes (always draw)
            for i in range(1, len(pts)):
                if pts[i-1] is None or pts[i] is None: continue
                cv2.line(canvas, pts[i-1], pts[i], 255, thickness=BRUSH_RADIUS, lineType=cv2.LINE_AA)

            # canvas preview
            preview = cv2.resize(canvas, (160,120))
            frame[10:10+120, W-10-160:W-10] = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

            # ensemble prediction
            nonzero = cv2.countNonZero(canvas)
            if thumbs_up and nonzero > MIN_STROKE_PIXELS and not pred_locked:
                mn = preprocess_canvas_to_mnist(canvas)
                if mn is not None:
                    inp = np.expand_dims(mn, 0).astype('float32')
                    probs_list = [m.predict(inp, verbose=0)[0] for m in models_list]
                    avg_probs = np.mean(probs_list, axis=0)
                    pred = int(np.argmax(avg_probs))
                    conf = float(np.max(avg_probs))
                    last_prediction = pred
                    last_conf = conf
                    pred_locked = True
                    prediction_time = time.time()
                    reset_until = time.time() + RESET_SECONDS
                    canvas[:] = 0
                    pts.clear()
                    smooth_buf.clear()
                    print(f"Ensemble Prediction: {pred} (conf {conf:.3f})")

            if pred_locked and (time.time() - prediction_time) > PRED_LOCK_SECONDS:
                pred_locked = False
                pts.clear()
                smooth_buf.clear()

            # overlays
            if last_prediction is not None:
                cv2.putText(frame, f"Prediction: {last_prediction}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
                cv2.putText(frame, f"Confidence: {last_conf:.2f}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            else:
                cv2.putText(frame, "Prediction: -", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)

            cv2.putText(frame, "Press 'c' to clear, 'q' to quit", (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

            cv2.imshow("MNIST Air Draw", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('c'): canvas[:] = 0; pts.clear(); smooth_buf.clear(); last_prediction=None; pred_locked=False

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Clean exit.")

# ----------------- Visualizations -----------------
def show_ensemble_performance(models_list):
    """Visualize test accuracy for each model and ensemble average."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = (x_test.astype('float32') / 255.0)[..., None]
    y_test_cat = to_categorical(y_test, 10)

    acc_list = []
    for i, m in enumerate(models_list):
        loss, acc = m.evaluate(x_test, y_test_cat, verbose=0)
        acc_list.append(acc)
        print(f"Model {i+1} test accuracy: {acc:.4f}")

    # Ensemble accuracy (mean probs)
    probs_all = np.mean([m.predict(x_test, verbose=0) for m in models_list], axis=0)
    y_pred_ensemble = np.argmax(probs_all, axis=1)
    ensemble_acc = np.mean(y_pred_ensemble == y_test)
    print(f"Ensemble test accuracy: {ensemble_acc:.4f}")

    # Plot accuracies
    plt.figure(figsize=(6,4))
    plt.bar([f"Model {i+1}" for i in range(len(models_list))] + ["Ensemble"],
            acc_list + [ensemble_acc], color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("Ensemble CNN Performance on MNIST")
    for i, v in enumerate(acc_list + [ensemble_acc]):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
    plt.show()

# ----------------- Main -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mnist_air_live_ensemble_visual.py")
    parser.add_argument("--retrain", action="store_true", help="Force retrain of CNNs even if saved models exist.")
    args = parser.parse_args()

    # Ask for writing hand
    while True:
        choice = input("Which is your writing hand? (L/R): ").strip().lower()
        if choice in ('l','r'):
            writing_hand = "Left" if choice == 'l' else "Right"
            break
        print("Enter 'L' or 'R'.")

    models_list = train_and_save_ensemble(num_models=NUM_MODELS, force_retrain=args.retrain)
    run_live_loop(models_list, writing_hand)
    show_ensemble_performance(models_list)
