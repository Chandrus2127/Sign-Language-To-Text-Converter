"""
=============================================================================
Sign Language To Text Conversion Application
=============================================================================
A real-time American Sign Language (ASL) finger-spelling recognition system
using deep convolutional neural networks with OpenCV and Tkinter.

Features:
- Live webcam feed with dynamic ROI extraction and bounding guide
- Real-time image preprocessing (grayscale, Gaussian blur, adaptive Otsu thresholding)
- Dynamic lighting / sensitivity slider to adapt to ambient brightness
- Two-tier CNN classification architecture:
    * Primary Classifier: 27 classes (Blank + A-Z)
    * Confusing Letter Disambiguation Classifiers:
        - DRU Model  : D, R, U
        - TKDI Model : T, K, D, I
        - SMN Model  : S, M, N
- Intelligent character stabilization and debouncing to prevent noisy frame duplicates
- Dynamic word formation with space, backspace, and clear controls
- Automatic sentence construction
- Context-aware word suggestions (prefix autocompletion + spellchecking)
- Assistive Text-to-Speech (TTS) audio vocalization
- Copy sentence to system clipboard
- In-app ASL Alphabet Reference Guide chart viewer
- Interactive Test / Demo Sign Mode using testing dataset samples
- Robust camera reconnection and graceful resource teardown
=============================================================================
"""

import os
import sys
import time
import operator
import string
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Import word suggestions engine
from common_words import get_word_suggestions


class SignLanguageApp:
    """
    Main application controller managing camera capture, preprocessing pipeline,
    neural network inference, debounced text generation, and Tkinter GUI.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("1040x880")
        self.root.minsize(1000, 840)
        self.root.configure(bg="#F1F5F9")
        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

        # ---------------- State Variables ----------------
        self.current_symbol = "-"
        self.word = ""
        self.sentence = ""
        self.suggestions = ["HELLO", "THANK YOU", "PLEASE"]
        self.is_paused = False
        self.vs = None
        self.loop_id = None
        self.threshold_val = 70

        # Demo / Test Mode state
        self.demo_mode = False
        self.demo_letter = "A"
        self.demo_samples = {}
        self._init_demo_dataset()

        # Stabilization / Debouncing state with Temporal EMA Smoothing
        self.candidate_char = None
        self.candidate_count = 0
        self.last_committed_char = None
        self.blank_frames = 0
        self.smoothed_probs = None
        self.alpha_ema = 0.35            # EMA smoothing factor for stability
        self.min_confidence = 0.40       # Reject weak noise below 40% confidence
        self.STABILIZE_FRAMES = 10       # ~0.45s with smoothed probabilities (fast & locked)
        self.BLANK_COMMIT_FRAMES = 22    # ~1.0 second of blank to auto-commit word

        # ---------------- Load ML Models ----------------
        self.load_models()

        # ---------------- Build GUI ----------------
        self.build_gui()

        # ---------------- Keyboard Shortcuts ----------------
        self.root.bind("<space>", lambda e: self.add_space())
        self.root.bind("<BackSpace>", lambda e: self.backspace())
        self.root.bind("<Escape>", lambda e: self.close_application())
        self.root.bind("<Control-s>", lambda e: self.speak_sentence())
        self.root.bind("<Control-S>", lambda e: self.speak_sentence())
        self.root.bind("<Control-c>", lambda e: self.copy_sentence())
        self.root.bind("<Control-C>", lambda e: self.copy_sentence())

        # ---------------- Initialize Camera ----------------
        self.initialize_camera()

        # ---------------- Start Video Loop ----------------
        self.video_loop()

    # =========================================================================
    # DEMO / TEST DATASET INITIALIZATION
    # =========================================================================

    def _init_demo_dataset(self):
        """Scans dataSet/testingData for available sample images for each letter."""
        self.demo_samples = {}
        test_dir = os.path.join("dataSet", "testingData")
        if os.path.exists(test_dir):
            for ch in string.ascii_uppercase:
                folder = os.path.join(test_dir, ch)
                if os.path.isdir(folder):
                    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
                    if files:
                        self.demo_samples[ch] = files

    def get_demo_sample_path(self, letter):
        """Returns the first sample image path for a given letter."""
        samples = self.demo_samples.get(letter.upper(), [])
        if samples:
            return samples[0]
        return None

    # =========================================================================
    # MODEL LOADING AND ARCHITECTURE RECONSTRUCTION
    # =========================================================================

    def _build_main_model_architecture(self):
        """Constructs the exact architecture matching Models/model_new."""
        model = Sequential([
            Input(shape=(128, 128, 1)),
            Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d"),
            MaxPooling2D(pool_size=(2, 2), name="max_pooling2d"),
            Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_1"),
            MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="dense"),
            Dense(128, activation="relu", name="dense_1"),
            Dropout(0.4, name="dropout"),
            Dense(96, activation="relu", name="dense_2"),
            Dropout(0.4, name="dropout_1"),
            Dense(64, activation="relu", name="dense_3"),
            Dense(27, activation="softmax", name="dense_4")
        ])
        return model

    def _build_sub_model_architecture(self, num_classes):
        """Constructs the exact architecture matching Models/model-bw_*."""
        model = Sequential([
            Input(shape=(128, 128, 1)),
            Conv2D(32, (3, 3), padding="valid", activation="relu", name="conv2d_1"),
            MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
            Conv2D(32, (3, 3), padding="valid", activation="relu", name="conv2d_2"),
            MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"),
            Conv2D(32, (3, 3), padding="valid", activation="relu", name="conv2d_3"),
            MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_3"),
            Flatten(name="flatten_1"),
            Dense(128, activation="relu", name="dense_1"),
            Dropout(0.4, name="dropout_1"),
            Dense(96, activation="relu", name="dense_2"),
            Dropout(0.4, name="dropout_2"),
            Dense(64, activation="relu", name="dense_3"),
            Dense(num_classes, activation="softmax", name="dense_4")
        ])
        return model

    def _load_single_model(self, h5_path, arch_builder):
        """
        Loads pre-trained model weights directly into the reconstructed Sequential
        architecture, avoiding legacy Keras 2 JSON schema incompatibilities.
        """
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Model file not found at: {h5_path}")

        model = arch_builder()
        model.load_weights(h5_path)
        return model

    def load_models(self):
        """Loads the primary ASL classifier and 3 specialized sub-classifiers."""
        print("[INFO] Loading trained Sign Language neural models...")
        try:
            self.model_main = self._load_single_model(
                os.path.join("Models", "model_new.h5"),
                self._build_main_model_architecture
            )

            self.model_dru = self._load_single_model(
                os.path.join("Models", "model-bw_dru.h5"),
                lambda: self._build_sub_model_architecture(3)
            )

            self.model_tkdi = self._load_single_model(
                os.path.join("Models", "model-bw_tkdi.h5"),
                lambda: self._build_sub_model_architecture(4)
            )

            self.model_smn = self._load_single_model(
                os.path.join("Models", "model-bw_smn.h5"),
                lambda: self._build_sub_model_architecture(3)
            )

            # Warmup models with dummy array
            dummy = np.zeros((1, 128, 128, 1), dtype=np.float32)
            self.model_main(dummy, training=False)
            self.model_dru(dummy, training=False)
            self.model_tkdi(dummy, training=False)
            self.model_smn(dummy, training=False)
            print("[INFO] All 4 models loaded and warmed up successfully.")
            self.models_loaded = True
        except Exception as e:
            self.models_loaded = False
            print(f"[ERROR] Failed to load models: {e}")
            messagebox.showerror(
                "Model Loading Error",
                f"Could not load pre-trained models:\n{e}\n\n"
                "Please verify that 'Models/model_new.h5', 'model-bw_dru.h5', "
                "'model-bw_tkdi.h5', and 'model-bw_smn.h5' exist."
            )

    # =========================================================================
    # WEBCAM INITIALIZATION & RECOVERY
    # =========================================================================

    def initialize_camera(self):
        """Attempts to initialize camera index 0 or index 1 gracefully."""
        if self.demo_mode:
            return True

        if self.vs is not None:
            try:
                self.vs.release()
            except Exception:
                pass
            self.vs = None

        print("[INFO] Initializing webcam...")
        for cam_index in (0, 1):
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.vs = cap
                    self.camera_index = cam_index
                    self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print(f"[INFO] Camera index {cam_index} connected successfully.")
                    if hasattr(self, "status_cam_label"):
                        self.status_cam_label.config(text=f"Camera: Online (Index {cam_index})", fg="#16A34A")
                    return True
                cap.release()

        print("[WARN] Webcam not accessible on index 0 or 1.")
        if hasattr(self, "status_cam_label"):
            self.status_cam_label.config(text="Camera: Offline / Not Detected", fg="#DC2626")
        return False

    def retry_camera(self):
        """Retry camera connection when requested by user."""
        if self.demo_mode:
            self.toggle_demo_mode()
        success = self.initialize_camera()
        if success:
            messagebox.showinfo("Camera Connected", "Webcam connected successfully!")
        else:
            messagebox.showwarning(
                "Camera Not Found",
                "Unable to connect to a webcam. You can also use '🎯 Test Sign Mode' to test signs from dataset."
            )

    # =========================================================================
    # IMAGE PREPROCESSING PIPELINE
    # =========================================================================

    def process_raw_roi(self, roi_bgr):
        """Processes a raw cropped ROI image with adaptive threshold and morphological denoising."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        ret, roi_processed = cv2.threshold(
            th3, self.threshold_val, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Morphological noise removal to eliminate speckle noise in background
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        roi_processed = cv2.morphologyEx(roi_processed, cv2.MORPH_OPEN, kernel)
        return roi_processed

    def preprocess_roi(self, frame):
        """
        Extracts Region of Interest (ROI) where user places hand and applies:
        1. Square boundary cropping (upper right)
        2. Grayscale conversion + Gaussian blur
        3. Adaptive thresholding + Otsu binarization + Morphological denoising
        4. Hand presence validation via contour analysis
        Returns:
            roi_raw       : Cropped BGR hand image
            roi_processed : Binary thresholded image
            coords        : (x1, y1, x2, y2) bounding box
            hand_detected : bool
        """
        h, w = frame.shape[:2]
        box_size = min(300, int(w * 0.48), int(h * 0.70))

        # Place ROI box in the upper-right corner of the mirrored view
        x2 = w - 16
        x1 = x2 - box_size
        y1 = 16
        y2 = y1 + box_size

        roi_raw = frame[y1:y2, x1:x2]
        if roi_raw is None or roi_raw.size == 0:
            return None, None, (x1, y1, x2, y2), False

        roi_processed = self.process_raw_roi(roi_raw)

        # Detect whether an actual hand is in the ROI (hand pixels are 0, background 255)
        hand_mask = cv2.bitwise_not(roi_processed)
        cnts, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_detected = False
        if cnts:
            max_c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(max_c) > (box_size * box_size * 0.04):
                hand_detected = True

        return roi_raw, roi_processed, (x1, y1, x2, y2), hand_detected

    def on_threshold_change(self, val):
        """Slider callback for adjusting lighting / binarization sensitivity."""
        try:
            self.threshold_val = int(val)
        except Exception:
            pass

    # =========================================================================
    # TWO-STAGE CLASSIFICATION PIPELINE (WITH TEMPORAL EMA SMOOTHING)
    # =========================================================================

    def predict_character(self, processed_roi, hand_detected=True):
        """
        Infers the ASL character from the processed ROI using:
        - Hand presence validation
        - Base classifier (27 classes)
        - Multi-frame temporal Exponential Moving Average (EMA) smoothing
        - Two-stage hierarchical disambiguation for visually similar letter groups
        Returns:
            predicted_char : str ('A'-'Z' or 'blank')
            confidence     : float
        """
        if not self.models_loaded or processed_roi is None or not hand_detected:
            # Gradually decay smoothed distribution when hand is withdrawn
            if self.smoothed_probs is not None:
                self.smoothed_probs *= 0.6
                if float(np.max(self.smoothed_probs)) < 0.2:
                    self.smoothed_probs = None
            return "blank", 0.0

        try:
            # Resize to model input specification: 128x128
            test_image = cv2.resize(processed_roi, (128, 128))
            input_tensor = test_image.reshape(1, 128, 128, 1).astype(np.float32)

            # Stage 1: Base Model Prediction
            raw_pred = self.model_main(input_tensor, training=False).numpy()[0]

            # Apply Temporal EMA Smoothing to eliminate single-frame glitches
            if self.smoothed_probs is None:
                self.smoothed_probs = raw_pred.copy()
            else:
                self.smoothed_probs = (
                    self.alpha_ema * raw_pred + (1.0 - self.alpha_ema) * self.smoothed_probs
                )

            pred_dict = {"blank": float(self.smoothed_probs[0])}
            for idx, ch in enumerate(string.ascii_uppercase):
                pred_dict[ch] = float(self.smoothed_probs[idx + 1])

            sorted_pred = sorted(pred_dict.items(), key=operator.itemgetter(1), reverse=True)
            candidate = sorted_pred[0][0]
            confidence = sorted_pred[0][1]

            # Gate weak predictions to prevent phantom characters
            if confidence < self.min_confidence:
                return "blank", confidence

            # Stage 2: Specialized Disambiguation
            # 2a. DRU Classifier
            if candidate in ("D", "R", "U"):
                res_dru = self.model_dru(input_tensor, training=False).numpy()[0]
                dru_dict = {
                    "D": float(res_dru[0]),
                    "R": float(res_dru[1]),
                    "U": float(res_dru[2])
                }
                candidate = max(dru_dict, key=dru_dict.get)
                confidence = dru_dict[candidate]

            # 2b. TKDI Classifier
            elif candidate in ("I", "K", "T"):
                res_tkdi = self.model_tkdi(input_tensor, training=False).numpy()[0]
                tkdi_dict = {
                    "D": float(res_tkdi[0]),
                    "I": float(res_tkdi[1]),
                    "K": float(res_tkdi[2]),
                    "T": float(res_tkdi[3])
                }
                candidate = max(tkdi_dict, key=tkdi_dict.get)
                confidence = tkdi_dict[candidate]

            # 2c. SMN Classifier
            elif candidate in ("M", "N", "S"):
                res_smn = self.model_smn(input_tensor, training=False).numpy()[0]
                smn_dict = {
                    "M": float(res_smn[0]),
                    "N": float(res_smn[1]),
                    "S": float(res_smn[2])
                }
                candidate = max(smn_dict, key=smn_dict.get)
                confidence = smn_dict[candidate]

            return candidate, confidence

        except Exception as e:
            print(f"[WARN] Error during inference: {e}")
            return "blank", 0.0

    # =========================================================================
    # PREDICTION STABILIZATION & DEBOUNCING
    # =========================================================================

    def stabilize_prediction(self, raw_char, confidence):
        """
        Stabilizes real-time predictions across consecutive video frames:
        - Requires a character to be held consistently for STABILIZE_FRAMES.
        - Prevents multiple additions of the same character in single gestures.
        - Automatically commits the word when hand is removed (blank) for BLANK_COMMIT_FRAMES.
        """
        # Case A: Blank gesture (empty ROI / hand relaxed)
        if raw_char == "blank" or raw_char is None:
            self.blank_frames += 1
            self.candidate_count = 0
            self.candidate_char = None
            self.current_symbol = "-"
            self.lbl_character.config(text="-")
            self.lbl_status.config(text="Waiting for gesture...", fg="#64748B")

            # If user withdrew hand for ~1 second, auto-commit word into sentence
            if self.blank_frames >= self.BLANK_COMMIT_FRAMES:
                if self.word.strip():
                    self.update_sentence()
                self.blank_frames = 0
                self.last_committed_char = None
            return

        # Case B: Letter gesture (A-Z)
        self.blank_frames = 0
        self.current_symbol = raw_char
        self.lbl_character.config(text=raw_char)

        if raw_char == self.candidate_char:
            self.candidate_count += 1
        else:
            self.candidate_char = raw_char
            self.candidate_count = 1

        # Check stabilization threshold
        if self.candidate_count >= self.STABILIZE_FRAMES:
            if self.candidate_char != self.last_committed_char:
                # Commit new recognized letter
                self.update_word(self.candidate_char)
                self.last_committed_char = self.candidate_char
                self.lbl_status.config(
                    text=f"[✓ Added '{self.candidate_char}'] (Conf: {confidence*100:.1f}%)", fg="#16A34A"
                )
            else:
                # Letter already committed; still held steady
                self.lbl_status.config(
                    text=f"[Held: '{self.candidate_char}'] (Conf: {confidence*100:.1f}%)", fg="#2563EB"
                )
        else:
            # Show visual hold progress indicator
            pct = int((self.candidate_count / self.STABILIZE_FRAMES) * 100)
            bars = "■" * (self.candidate_count * 6 // self.STABILIZE_FRAMES)
            self.lbl_status.config(
                text=f"Holding [{bars:<6}] {pct}% (Conf: {confidence*100:.1f}%)", fg="#D97706"
            )

    # =========================================================================
    # WORD & SENTENCE FORMATION LOGIC
    # =========================================================================

    def update_word(self, char):
        """Appends a confirmed character to the active word."""
        self.word += char
        self.lbl_word.config(text=self.word)
        self.generate_suggestions()

    def update_sentence(self):
        """Commits the active word into the sentence."""
        w = self.word.strip()
        if w:
            if self.sentence:
                self.sentence += " " + w
            else:
                self.sentence = w
            self.lbl_sentence.config(text=self.sentence)

        self.word = ""
        self.lbl_word.config(text="")
        self.candidate_char = None
        self.candidate_count = 0
        self.generate_suggestions()

    def add_space(self):
        """SPACE button handler: completes the current word and appends to sentence."""
        self.update_sentence()

    def backspace(self):
        """BACKSPACE button handler: removes last character from word or sentence."""
        if self.word:
            self.word = self.word[:-1]
            self.lbl_word.config(text=self.word)
            self.generate_suggestions()
        elif self.sentence:
            # Remove last character from sentence
            self.sentence = self.sentence[:-1].rstrip()
            self.lbl_sentence.config(text=self.sentence)
        self.last_committed_char = None

    def clear_word(self):
        """CLEAR WORD button handler: clears current word and resets buffer."""
        self.word = ""
        self.lbl_word.config(text="")
        self.candidate_char = None
        self.candidate_count = 0
        self.last_committed_char = None
        self.generate_suggestions()

    def clear_sentence(self):
        """CLEAR SENTENCE button handler: clears entire formed sentence."""
        self.sentence = ""
        self.lbl_sentence.config(text="")

    def clear_all(self):
        """Clears both active word and sentence."""
        self.clear_word()
        self.clear_sentence()

    # =========================================================================
    # ASSISTIVE FEATURES: TTS & CLIPBOARD
    # =========================================================================

    def speak_sentence(self):
        """Vocalizes the current formed sentence or word using Windows SAPI TTS."""
        target_text = self.sentence.strip() or self.word.strip()
        if not target_text:
            self.lbl_status.config(text="[No text to speak]", fg="#D97706")
            return

        self.lbl_status.config(text=f"[🔊 Speaking: '{target_text}']", fg="#7C3AED")

        def _worker():
            try:
                # Windows built-in SAPI via PowerShell System.Speech
                safe_text = target_text.replace("'", "''").replace('"', '`"')
                cmd = f'powershell -WindowStyle Hidden -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{safe_text}\')"'
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[WARN] TTS error: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def copy_sentence(self):
        """Copies active sentence or word to the system clipboard."""
        target_text = self.sentence.strip() or self.word.strip()
        if target_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(target_text)
            self.root.update()
            self.lbl_status.config(text="[✓ Copied to Clipboard]", fg="#16A34A")
        else:
            self.lbl_status.config(text="[Clipboard: buffer empty]", fg="#64748B")

    # =========================================================================
    # ASL REFERENCE CHART VIEWER
    # =========================================================================

    def show_sign_chart(self):
        """Displays the ASL fingerspelling reference chart in a top-level window."""
        chart_path = os.path.join("images", "signs.jpg")
        if not os.path.exists(chart_path):
            messagebox.showwarning("File Missing", "Sign reference chart image ('images/signs.jpg') not found.")
            return

        top = tk.Toplevel(self.root)
        top.title("ASL Fingerspelling Reference Guide (A - Z)")
        top.geometry("740x620")
        top.configure(bg="#F8FAFC")

        lbl_header = tk.Label(
            top,
            text="American Sign Language (ASL) Alphabet Chart",
            font=("Segoe UI", 14, "bold"),
            bg="#F8FAFC",
            fg="#0F172A",
            pady=8
        )
        lbl_header.pack()

        try:
            chart_img = Image.open(chart_path)
            chart_img = chart_img.resize((700, 520), Image.Resampling.LANCZOS)
            chart_photo = ImageTk.PhotoImage(chart_img)
            lbl_img = tk.Label(top, image=chart_photo, bg="#FFFFFF", bd=1, relief="solid")
            lbl_img.image = chart_photo  # Keep reference
            lbl_img.pack(padx=10, pady=5)
        except Exception as e:
            tk.Label(top, text=f"Failed to load image: {e}", fg="red").pack(pady=20)

    # =========================================================================
    # WORD SUGGESTIONS LOGIC
    # =========================================================================

    def generate_suggestions(self):
        """Queries the suggestion engine for 3 smart word suggestions."""
        self.suggestions = get_word_suggestions(self.word, max_count=3)
        for i in range(3):
            sug_text = self.suggestions[i] if i < len(self.suggestions) else ""
            btn = self.sug_buttons[i]
            if sug_text:
                btn.config(text=sug_text, state=tk.NORMAL)
            else:
                btn.config(text="-", state=tk.DISABLED)

    def apply_suggestion(self, index):
        """Clicking a suggestion replaces the active word with that suggestion."""
        if 0 <= index < len(self.suggestions):
            picked = self.suggestions[index]
            if picked and picked != "-":
                self.word = picked
                self.lbl_word.config(text=self.word)
                self.last_committed_char = None
                self.generate_suggestions()

    # =========================================================================
    # CAMERA / PAUSE / DEMO CONTROLS
    # =========================================================================

    def toggle_pause(self):
        """Toggles real-time webcam feed pause state."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶ RESUME", bg="#059669")
            self.lbl_status.config(text="[FEED PAUSED]", fg="#7C3AED")
        else:
            self.btn_pause.config(text="⏸ PAUSE", bg="#7C3AED")
            self.lbl_status.config(text="Active", fg="#16A34A")

    def toggle_demo_mode(self):
        """Toggles between Live Camera Feed and Test Dataset Sign Mode."""
        self.demo_mode = not self.demo_mode
        if self.demo_mode:
            self.btn_demo.config(text="🎯 TEST SIGN: ON", bg="#D97706")
            self.cb_demo_letter.config(state="readonly")
            self.status_cam_label.config(text="Mode: Test Sign (Dataset Sample)", fg="#D97706")
            if self.vs is not None:
                try:
                    self.vs.release()
                except Exception:
                    pass
                self.vs = None
        else:
            self.btn_demo.config(text="🎯 TEST SIGN: OFF", bg="#0D9488")
            self.cb_demo_letter.config(state="disabled")
            self.initialize_camera()

    def on_demo_letter_select(self, event=None):
        """Handles user selecting a test sign letter from the combobox."""
        selected = self.cb_demo_letter.get()
        if selected:
            self.demo_letter = selected
            self.candidate_char = None
            self.candidate_count = 0
            self.last_committed_char = None

    # =========================================================================
    # VIDEO UPDATE LOOP
    # =========================================================================

    def video_loop(self):
        """Main real-time loop capturing frames, preprocessing, and rendering."""
        # Branch 1: Test Sign Dataset Mode
        if self.demo_mode:
            sample_path = self.get_demo_sample_path(self.demo_letter)
            if sample_path and os.path.exists(sample_path):
                raw_roi = cv2.imread(sample_path)
                if raw_roi is not None:
                    # Preprocess with active threshold
                    roi_processed = self.process_raw_roi(raw_roi)
                    if not self.is_paused:
                        raw_char, confidence = self.predict_character(roi_processed)
                        self.stabilize_prediction(raw_char, confidence)

                    # Build simulated frame card showing raw sign
                    demo_frame = np.full((345, 460, 3), (241, 245, 249), dtype=np.uint8)
                    resized_raw = cv2.resize(raw_roi, (260, 260))
                    demo_frame[42:302, 100:360] = resized_raw
                    cv2.rectangle(demo_frame, (98, 40), (362, 304), (255, 120, 0), 2)
                    cv2.putText(
                        demo_frame,
                        f"DEMO DATASET SIGN: '{self.demo_letter}'",
                        (20, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (15, 23, 42),
                        2
                    )

                    # Render left panel
                    frame_rgb = cv2.cvtColor(demo_frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    imgtk_webcam = ImageTk.PhotoImage(image=frame_pil)
                    self.panel_webcam.imgtk = imgtk_webcam
                    self.panel_webcam.config(image=imgtk_webcam)

                    # Render right panel
                    roi_pil = Image.fromarray(roi_processed).resize((300, 300), Image.Resampling.NEAREST)
                    imgtk_roi = ImageTk.PhotoImage(image=roi_pil)
                    self.panel_roi.imgtk = imgtk_roi
                    self.panel_roi.config(image=imgtk_roi)

        # Branch 2: Live Webcam Feed
        elif not self.is_paused and self.vs is not None and self.vs.isOpened():
            ok, frame = self.vs.read()

            if ok and frame is not None:
                # Mirror image horizontally so gestures mirror natural movements
                flipped = cv2.flip(frame, 1)

                # Preprocess hand ROI with contour validation
                roi_raw, roi_processed, (x1, y1, x2, y2), hand_detected = self.preprocess_roi(flipped)

                # Predict gesture from binary ROI with EMA smoothing
                if roi_processed is not None:
                    raw_char, confidence = self.predict_character(roi_processed, hand_detected=hand_detected)
                    self.stabilize_prediction(raw_char, confidence)

                # Draw clean dynamic bounding box and status label on webcam display feed
                display_frame = flipped.copy()
                box_color = (0, 200, 100) if hand_detected else (255, 120, 0)
                box_label = "Hand Ready [Aligned]" if hand_detected else "Place Hand Inside Box"
                cv2.rectangle(display_frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), box_color, 2)
                cv2.putText(
                    display_frame,
                    box_label,
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2
                )

                # Render Webcam Feed (Left Panel)
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb).resize((460, 345), Image.Resampling.BILINEAR)
                imgtk_webcam = ImageTk.PhotoImage(image=frame_pil)
                self.panel_webcam.imgtk = imgtk_webcam
                self.panel_webcam.config(image=imgtk_webcam)

                # Render Processed ROI (Right Panel)
                if roi_processed is not None:
                    roi_pil = Image.fromarray(roi_processed).resize((300, 300), Image.Resampling.NEAREST)
                    imgtk_roi = ImageTk.PhotoImage(image=roi_pil)
                    self.panel_roi.imgtk = imgtk_roi
                    self.panel_roi.config(image=imgtk_roi)

        elif self.vs is None or not self.vs.isOpened():
            # Show camera disconnected placeholder
            self._render_camera_offline()

        # Schedule next frame (~30 FPS)
        self.loop_id = self.root.after(30, self.video_loop)

    def _render_camera_offline(self):
        """Displays offline banner if webcam is detached."""
        offline_img = np.zeros((345, 460, 3), dtype=np.uint8)
        offline_img[:] = (30, 41, 59)
        cv2.putText(
            offline_img,
            "Webcam Offline / Not Found",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )
        cv2.putText(
            offline_img,
            "Click 'Retry Camera' or toggle 'Test Sign Mode'",
            (35, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (148, 163, 184),
            1
        )
        pil_img = Image.fromarray(offline_img)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.panel_webcam.imgtk = imgtk
        self.panel_webcam.config(image=imgtk)

    # =========================================================================
    # GUI LAYOUT DESIGN
    # =========================================================================

    def build_gui(self):
        """Constructs the application interface matching modern design standards."""

        # Outer Container
        main_container = tk.Frame(self.root, bg="#F1F5F9")
        main_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        # ---------------- Title Header ----------------
        header_frame = tk.Frame(main_container, bg="#F1F5F9")
        header_frame.pack(fill=tk.X, pady=(0, 6))

        title_label = tk.Label(
            header_frame,
            text="Sign Language To Text Conversion",
            font=("Courier", 24, "bold"),
            fg="#0F172A",
            bg="#F1F5F9"
        )
        title_label.pack()

        # ---------------- Video Panels Row (Webcam on Left, ROI on Right) ----------------
        video_row = tk.Frame(main_container, bg="#F1F5F9")
        video_row.pack(fill=tk.X, pady=(0, 8))

        # Left Column: Webcam Frame Card
        cam_card = tk.Frame(video_row, bg="#FFFFFF", bd=1, relief="solid")
        cam_card.pack(side=tk.LEFT, padx=(0, 10), fill=tk.BOTH, expand=True)

        cam_header = tk.Label(
            cam_card,
            text="LIVE WEBCAM FEED (PLACE HAND IN BOX)",
            font=("Segoe UI", 9, "bold"),
            fg="#475569",
            bg="#F8FAFC",
            pady=3
        )
        cam_header.pack(fill=tk.X)

        self.panel_webcam = tk.Label(cam_card, bg="#0F172A", width=460, height=345)
        self.panel_webcam.pack(padx=6, pady=6)

        # Right Column: Processed Hand ROI Card
        roi_card = tk.Frame(video_row, bg="#FFFFFF", bd=1, relief="solid")
        roi_card.pack(side=tk.RIGHT, padx=(10, 0), fill=tk.BOTH, expand=False)

        roi_header = tk.Label(
            roi_card,
            text="PROCESSED HAND IMAGE (ROI)",
            font=("Segoe UI", 9, "bold"),
            fg="#475569",
            bg="#F8FAFC",
            pady=3
        )
        roi_header.pack(fill=tk.X)

        self.panel_roi = tk.Label(roi_card, bg="#0F172A", width=300, height=300)
        self.panel_roi.pack(padx=6, pady=(6, 6))

        # ---------------- Information Display Area ----------------
        info_frame = tk.Frame(main_container, bg="#FFFFFF", bd=1, relief="solid", padx=16, pady=8)
        info_frame.pack(fill=tk.X, pady=(0, 8))

        # Row 1: Character
        char_row = tk.Frame(info_frame, bg="#FFFFFF")
        char_row.pack(fill=tk.X, pady=1)

        lbl_c_tag = tk.Label(
            char_row,
            text="Character :",
            font=("Courier", 20, "bold"),
            fg="#0F172A",
            bg="#FFFFFF",
            width=12,
            anchor="w"
        )
        lbl_c_tag.pack(side=tk.LEFT)

        self.lbl_character = tk.Label(
            char_row,
            text="-",
            font=("Courier", 22, "bold"),
            fg="#1D4ED8",
            bg="#FFFFFF",
            width=3,
            anchor="w"
        )
        self.lbl_character.pack(side=tk.LEFT)

        self.lbl_status = tk.Label(
            char_row,
            text="Ready",
            font=("Segoe UI", 10, "bold"),
            fg="#64748B",
            bg="#FFFFFF"
        )
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        # Row 2: Word
        word_row = tk.Frame(info_frame, bg="#FFFFFF")
        word_row.pack(fill=tk.X, pady=1)

        lbl_w_tag = tk.Label(
            word_row,
            text="Word      :",
            font=("Courier", 20, "bold"),
            fg="#0F172A",
            bg="#FFFFFF",
            width=12,
            anchor="w"
        )
        lbl_w_tag.pack(side=tk.LEFT)

        self.lbl_word = tk.Label(
            word_row,
            text="",
            font=("Courier", 22, "bold"),
            fg="#0F172A",
            bg="#FFFFFF",
            anchor="w"
        )
        self.lbl_word.pack(side=tk.LEFT)

        # Row 3: Sentence
        sent_row = tk.Frame(info_frame, bg="#FFFFFF")
        sent_row.pack(fill=tk.X, pady=1)

        lbl_s_tag = tk.Label(
            sent_row,
            text="Sentence  :",
            font=("Courier", 20, "bold"),
            fg="#0F172A",
            bg="#FFFFFF",
            width=12,
            anchor="w"
        )
        lbl_s_tag.pack(side=tk.LEFT)

        self.lbl_sentence = tk.Label(
            sent_row,
            text="",
            font=("Courier", 18, "bold"),
            fg="#0F172A",
            bg="#FFFFFF",
            anchor="w",
            wraplength=720,
            justify=tk.LEFT
        )
        self.lbl_sentence.pack(side=tk.LEFT)

        # ---------------- Suggestions Section ----------------
        sug_container = tk.Frame(main_container, bg="#F1F5F9")
        sug_container.pack(fill=tk.X, pady=(0, 6))

        sug_title = tk.Label(
            sug_container,
            text="Suggestions :",
            font=("Courier", 18, "bold"),
            fg="#DC2626",
            bg="#F1F5F9"
        )
        sug_title.pack(anchor="center", pady=(0, 4))

        # 3 Suggestion Buttons
        buttons_row = tk.Frame(sug_container, bg="#F1F5F9")
        buttons_row.pack(anchor="center")

        self.sug_buttons = []
        for i in range(3):
            btn = tk.Button(
                buttons_row,
                text=self.suggestions[i],
                font=("Courier", 14, "bold"),
                fg="#0F172A",
                bg="#FFFFFF",
                activebackground="#E2E8F0",
                bd=2,
                relief="groove",
                width=15,
                cursor="hand2",
                command=lambda idx=i: self.apply_suggestion(idx)
            )
            btn.pack(side=tk.LEFT, padx=10)
            self.sug_buttons.append(btn)

        # ---------------- Action / Control Toolbar ----------------
        # Row 1: Editing & Assistive Buttons
        toolbar_row1 = tk.Frame(main_container, bg="#F1F5F9")
        toolbar_row1.pack(fill=tk.X, pady=(2, 4))

        btn_row1_inner = tk.Frame(toolbar_row1, bg="#F1F5F9")
        btn_row1_inner.pack(anchor="center")

        btn_style_main = {
            "font": ("Segoe UI", 9, "bold"),
            "padx": 10,
            "pady": 5,
            "bd": 0,
            "cursor": "hand2"
        }

        # SPACE Button
        btn_space = tk.Button(
            btn_row1_inner,
            text="␣ SPACE",
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            command=self.add_space,
            **btn_style_main
        )
        btn_space.pack(side=tk.LEFT, padx=4)

        # BACKSPACE Button
        btn_backspace = tk.Button(
            btn_row1_inner,
            text="⌫ BACKSPACE",
            bg="#D97706",
            fg="#FFFFFF",
            activebackground="#B45309",
            activeforeground="#FFFFFF",
            command=self.backspace,
            **btn_style_main
        )
        btn_backspace.pack(side=tk.LEFT, padx=4)

        # SPEAK Button
        btn_speak = tk.Button(
            btn_row1_inner,
            text="🔊 SPEAK",
            bg="#7C3AED",
            fg="#FFFFFF",
            activebackground="#6D28D9",
            activeforeground="#FFFFFF",
            command=self.speak_sentence,
            **btn_style_main
        )
        btn_speak.pack(side=tk.LEFT, padx=4)

        # COPY Button
        btn_copy = tk.Button(
            btn_row1_inner,
            text="📋 COPY",
            bg="#059669",
            fg="#FFFFFF",
            activebackground="#047857",
            activeforeground="#FFFFFF",
            command=self.copy_sentence,
            **btn_style_main
        )
        btn_copy.pack(side=tk.LEFT, padx=4)

        # CLEAR WORD Button
        btn_clear_w = tk.Button(
            btn_row1_inner,
            text="✕ CLEAR WORD",
            bg="#64748B",
            fg="#FFFFFF",
            activebackground="#475569",
            activeforeground="#FFFFFF",
            command=self.clear_word,
            **btn_style_main
        )
        btn_clear_w.pack(side=tk.LEFT, padx=4)

        # CLEAR ALL Button
        btn_clear_all = tk.Button(
            btn_row1_inner,
            text="🗑 CLEAR ALL",
            bg="#DC2626",
            fg="#FFFFFF",
            activebackground="#B91C1C",
            activeforeground="#FFFFFF",
            command=self.clear_all,
            **btn_style_main
        )
        btn_clear_all.pack(side=tk.LEFT, padx=4)

        # Row 2: Secondary Controls (Sign Chart, Demo Mode, Threshold Slider, Pause, Camera, Exit)
        toolbar_row2 = tk.Frame(main_container, bg="#F1F5F9")
        toolbar_row2.pack(fill=tk.X, pady=(2, 4))

        btn_row2_inner = tk.Frame(toolbar_row2, bg="#F1F5F9")
        btn_row2_inner.pack(anchor="center")

        # SIGN CHART Button
        btn_chart = tk.Button(
            btn_row2_inner,
            text="📖 SIGN CHART",
            bg="#4F46E5",
            fg="#FFFFFF",
            activebackground="#4338CA",
            activeforeground="#FFFFFF",
            command=self.show_sign_chart,
            **btn_style_main
        )
        btn_chart.pack(side=tk.LEFT, padx=4)

        # DEMO / TEST SIGN Button
        self.btn_demo = tk.Button(
            btn_row2_inner,
            text="🎯 TEST SIGN: OFF",
            bg="#0D9488",
            fg="#FFFFFF",
            activebackground="#0F766E",
            activeforeground="#FFFFFF",
            command=self.toggle_demo_mode,
            **btn_style_main
        )
        self.btn_demo.pack(side=tk.LEFT, padx=4)

        # DEMO LETTER COMBOBOX
        lbl_sign_pick = tk.Label(btn_row2_inner, text="Sign:", font=("Segoe UI", 9, "bold"), bg="#F1F5F9", fg="#334155")
        lbl_sign_pick.pack(side=tk.LEFT, padx=(4, 2))

        self.cb_demo_letter = ttk.Combobox(
            btn_row2_inner,
            values=list(string.ascii_uppercase),
            width=3,
            state="disabled",
            font=("Segoe UI", 9, "bold")
        )
        self.cb_demo_letter.set("A")
        self.cb_demo_letter.bind("<<ComboboxSelected>>", self.on_demo_letter_select)
        self.cb_demo_letter.pack(side=tk.LEFT, padx=(0, 6))

        # LIGHTING / THRESHOLD SLIDER
        lbl_slider = tk.Label(btn_row2_inner, text="Sensitivity:", font=("Segoe UI", 9, "bold"), bg="#F1F5F9", fg="#334155")
        lbl_slider.pack(side=tk.LEFT, padx=(4, 2))

        self.scale_thresh = tk.Scale(
            btn_row2_inner,
            from_=30,
            to=120,
            orient=tk.HORIZONTAL,
            length=100,
            showvalue=True,
            bg="#F1F5F9",
            highlightthickness=0,
            command=self.on_threshold_change
        )
        self.scale_thresh.set(70)
        self.scale_thresh.pack(side=tk.LEFT, padx=(0, 8))

        # PAUSE Button
        self.btn_pause = tk.Button(
            btn_row2_inner,
            text="⏸ PAUSE",
            bg="#7C3AED",
            fg="#FFFFFF",
            activebackground="#6D28D9",
            activeforeground="#FFFFFF",
            command=self.toggle_pause,
            **btn_style_main
        )
        self.btn_pause.pack(side=tk.LEFT, padx=4)

        # RETRY CAMERA Button
        btn_retry = tk.Button(
            btn_row2_inner,
            text="🔄 RETRY CAM",
            bg="#0D9488",
            fg="#FFFFFF",
            activebackground="#0F766E",
            activeforeground="#FFFFFF",
            command=self.retry_camera,
            **btn_style_main
        )
        btn_retry.pack(side=tk.LEFT, padx=4)

        # EXIT Button
        btn_exit = tk.Button(
            btn_row2_inner,
            text="⏹ EXIT",
            bg="#334155",
            fg="#FFFFFF",
            activebackground="#1E293B",
            activeforeground="#FFFFFF",
            command=self.close_application,
            **btn_style_main
        )
        btn_exit.pack(side=tk.LEFT, padx=4)

        # ---------------- Status Footer Bar ----------------
        status_bar = tk.Frame(self.root, bg="#E2E8F0", bd=1, relief="sunken", padx=10, pady=3)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_cam_label = tk.Label(
            status_bar,
            text="Camera: Initializing...",
            font=("Segoe UI", 9),
            bg="#E2E8F0",
            fg="#334155"
        )
        self.status_cam_label.pack(side=tk.LEFT, padx=6)

        models_status = "Models: 4/4 Loaded" if self.models_loaded else "Models: Load Failed"
        status_model_label = tk.Label(
            status_bar,
            text=models_status,
            font=("Segoe UI", 9),
            bg="#E2E8F0",
            fg="#16A34A" if self.models_loaded else "#DC2626"
        )
        status_model_label.pack(side=tk.LEFT, padx=12)

        hint_label = tk.Label(
            status_bar,
            text="Space: Add Word | Backspace: Del | Ctrl+S: Speak | Ctrl+C: Copy | Esc: Exit",
            font=("Segoe UI", 9, "italic"),
            bg="#E2E8F0",
            fg="#475569"
        )
        hint_label.pack(side=tk.RIGHT, padx=6)

    # =========================================================================
    # APPLICATION TEARDOWN
    # =========================================================================

    def close_application(self):
        """Safely stops timers, releases webcam resource, and closes Tkinter."""
        print("[INFO] Closing Sign Language To Text Conversion Application...")
        if self.loop_id is not None:
            try:
                self.root.after_cancel(self.loop_id)
            except Exception:
                pass
            self.loop_id = None

        if self.vs is not None:
            try:
                self.vs.release()
            except Exception:
                pass
            self.vs = None

        cv2.destroyAllWindows()
        try:
            self.root.destroy()
        except Exception:
            pass


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print(" Starting Sign Language To Text Conversion Application...")
    print("=" * 65)
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()