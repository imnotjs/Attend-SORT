# ==============================================================================
# GPU RUNTIME INITIALIZATION LAYER (COMPREHENSIVE DEPENDENCY LINKER)
# Registers absolute execution bins to completely eliminate downstream Error 126.
# ==============================================================================
import os
import sys

_conda_base = os.path.dirname(sys.executable)
_site_packages = os.path.join(_conda_base, "Lib", "site-packages")

# 1. Kumpulkan semua path target (eksplisit & hasil scanning otomatis)
_dll_paths = [
    os.path.join(_site_packages, "onnxruntime", "capi"),
    os.path.join(_site_packages, "nvidia", "cublas", "bin"),
    os.path.join(_site_packages, "nvidia", "cuda_runtime", "bin"),
    os.path.join(_site_packages, "nvidia", "cuda_nvrtc", "bin"),
    os.path.join(_site_packages, "nvidia", "cudnn", "bin"),
    os.path.join(_site_packages, "nvidia", "cufft", "bin"),
    os.path.join(_site_packages, "nvidia", "curand", "bin")
]

_nvidia_root = os.path.join(_site_packages, "nvidia")
if os.path.exists(_nvidia_root):
    for _root, _dirs, _ in os.walk(_nvidia_root):
        if "bin" in _dirs:
            _dll_paths.append(os.path.join(_root, "bin"))

# 2. Daftarkan semua path unik yang valid ke Windows DLL loader & PATH Environment
for _path in sorted(list(set(_dll_paths))):
    if os.path.exists(_path):
        os.add_dll_directory(_path)
        os.environ["PATH"] = _path + os.path.pathsep + os.environ["PATH"]

import cv2
import numpy as np
import time
import threading
import queue
import av
from datetime import datetime

from face_db_manager import FaceDBManager
from identity_matcher import IdentityMatcher
from res_opt_engine import ResOptEngine
from ultralytics import YOLO

# ==============================================================================
# CONFIGURASI DIREKTORI & PATH
# ==============================================================================
BASE_DIR = r"C:\Projects\FaceSORT Live Demo"
FACE_DB_DIR = os.path.join(BASE_DIR, "Face Database")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "widerperson_best.pt")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "live_demo_recorded.mp4")
ANOMALY_DIR = os.path.join(BASE_DIR, "Anomaly_Screenshots")

os.makedirs(ANOMALY_DIR, exist_ok=True)

# ==============================================================================
# STARTUP DIAGNOSTIC: Cek ONNX Runtime GPU availability
# Ini sumber bottleneck terbesar jika InsightFace berjalan di CPU.
# Error 126 (LoadLibrary failed) = onnxruntime-gpu tidak terinstall atau
# versi CUDA toolkit tidak cocok dengan onnxruntime-gpu yang diinstall.
#
# Cara fix (jalankan di miniconda env yang sama):
#   pip uninstall onnxruntime -y
#   pip install onnxruntime-gpu
#   (Pastikan CUDA toolkit >= 11.8 terinstall di sistem)
#
# Kalau sudah install onnxruntime-gpu tapi masih error 126:
#   Kemungkinan: cuDNN DLL tidak ada di PATH.
#   Fix: pip install nvidia-cudnn-cu11 (atau cu12 sesuai CUDA versi)
#        lalu tambah folder site-packages/nvidia/cudnn/bin ke PATH.
# ==============================================================================
try:
    import onnxruntime as _ort
    _ort_providers = _ort.get_available_providers()
    _ort_has_cuda = 'CUDAExecutionProvider' in _ort_providers
    _ort_version = _ort.__version__
    if _ort_has_cuda:
        print(f"✅ [ONNX RT] GPU provider tersedia (v{_ort_version}). InsightFace akan pakai GPU.")
    else:
        print(f"⚠️  [ONNX RT] CUDAExecutionProvider TIDAK tersedia (v{_ort_version})!")
        print(f"   Available providers: {_ort_providers}")
        print(f"   → InsightFace akan berjalan di CPU — face det ~65ms/frame bukan ~15ms.")
        print(f"   → FIX: pip uninstall onnxruntime -y && pip install onnxruntime-gpu")
        print(f"   → Pastikan CUDA toolkit >= 11.8 terinstall: nvidia-smi untuk cek versi.")
except ImportError:
    print(f"⚠️  [ONNX RT] onnxruntime tidak terinstall sama sekali!")

# ---------------------------------------------------------------------------
# shared_display_boxes sekarang menyimpan koordinat SUDAH dalam space 1280x720.
# AI thread scale koordinat sebelum publish, GUI thread tidak perlu scale lagi.
# Eliminasi satu sumber lag: dulu GUI thread re-scale tiap render dari data stale.
# ---------------------------------------------------------------------------
shared_display_boxes = []
shared_boxes_lock = threading.Lock()
fps_inf_calc = 0.0
attendance_final_log = {}
track_first_seen_frame = {}
unknown_screenshot_buffer = {}

# ==============================================================================
# FIX STATE LOSS: Global NRP-level verified state, independent dari track_id.
#
# Root cause bug "Tidak dikenali pas mendekat":
# Ketika orang mendekati kamera, bbox berubah drastis (scale naik 3-5x).
# BoTSort bisa kehilangan track dan assign track_id BARU → semua state di
# priority_registry (status_absen, nrp_name) hilang karena keyed by track_id.
# Track baru = register_new_track = priority=HIGH = "SEARCHING..." lagi.
#
# Solusi: verified_nrp_state adalah dict {nrp: True} yang TIDAK pernah di-reset
# selama session. Setiap kali track baru muncul dengan body Re-ID match ke NRP
# yang sudah verified, dia langsung di-set LOW priority + inherit nama.
# ==============================================================================
verified_nrp_state = {}  # { nrp_id: True } — persists across all track_id changes

# ==============================================================================
# FIX TTFM MEASUREMENT: track_first_seen_time pakai wall-clock time.time().
#
# Bug lama: TTFM dihitung dari frame count × (1/60).
# Asumsi ini salah karena AI thread TIDAK selalu jalan di 60fps — bisa 20-40fps
# tergantung load GPU. Akibatnya angka TTFM di laporan bisa meleset 1.5–3x
# dari durasi yang orang actually tunggu di dunia nyata.
#
# Fix: catat time.time() saat track_id pertama kali muncul.
# Saat match berhasil, TTFM = time.time() - track_first_seen_time[track_id].
# Ini akurat terlepas dari AI inference rate.
# ==============================================================================
track_first_seen_time = {}  # { track_id: float (epoch seconds) }


# ==============================================================================
# THREAD 1: HARDWARE CAMERA CAPTURE THREAD
# ==============================================================================
class CameraCaptureThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)

        # ==============================================================================
        # CAMERA CONFIG — NYK Nemesis A96 Severus
        #
        # Hardware constraint kamera ini:
        #   2560×1920 (2K) → max 30 FPS  ← OpenCV default kalau tidak di-set manual
        #   1920×1080      → max 60 FPS  ✅ ← ini yang kita target
        #   1280×720       → max 60 FPS  ✅
        #
        # WAJIB set FOURCC ke MJPG dulu SEBELUM set resolusi & FPS.
        # Kalau urutan terbalik atau FOURCC tidak di-set, driver DirectShow
        # fallback ke YUY2 yang bandwidth-nya terlalu besar untuk USB 2.0 di 60fps
        # → driver auto-turunkan FPS ke 15 atau 30 tanpa error apapun.
        #
        # Urutan yang benar: FOURCC → WIDTH → HEIGHT → FPS
        # ==============================================================================
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 50)

        # Verifikasi: driver tidak selalu honor nilai yang di-set.
        # Baca kembali nilai aktual yang dikonfirmasi driver DirectShow.
        actual_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"📷 [Camera] Negotiated mode : {actual_w}×{actual_h} @ {actual_fps:.0f} FPS")
        if actual_fps < 45:
            print(f"⚠️  [Camera] WARNING: Driver hanya kasih {actual_fps:.0f} FPS — bukan 50!")
            print(f"   Kemungkinan penyebab: port USB 2.0, kabel ekstensi, atau resolusi terlalu tinggi.")
            print(f"   Coba pindah ke port USB 3.0 atau turunkan resolusi ke 1280×720.")
        else:
            print(f"✅ [Camera] 50 FPS confirmed by driver.")

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self._frame_id = 0  # increment setiap frame baru dari hardware

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                    self._frame_id += 1
            else:
                time.sleep(0.001)

    def read_latest_frame(self):
        with self.lock:
            if self.frame is None:
                return False, None, -1
            return self.ret, self.frame.copy(), self._frame_id

    def stop(self):
        self.running = False
        self.cap.release()


# ==============================================================================
# THREAD 2: ASYNCHRONOUS VIDEO RECORDER
# ==============================================================================
import av

class HardwareVideoRecorder:
    def __init__(self, output_path, fps, resolution):
        self.output_path = output_path
        self.fps = fps
        self.width, self.height = resolution
        self.frame_queue = queue.Queue(maxsize=120)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.write_frames, args=())
        t.daemon = True
        t.start()
        return self

    def write_frame_async(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())

    def write_frames(self):
        # Buka container video
        container = av.open(self.output_path, mode='w')
        stream = None
        
        # Buat satu frame dummy kecil (hitam) untuk tes inisialisasi hardware encoder
        test_frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format='rgb24')

        # 1. Coba inisialisasi menggunakan NVIDIA NVENC
        try:
            stream = container.add_stream('h264_nvenc', rate=int(self.fps))
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = 'yuv420p'
            
            # Paksa encoder untuk membuka jalur hardware (Warm-up Test)
            list(stream.encode(test_frame))
            print("🚀 [Recorder] Berhasil mengunci NVIDIA NVENC Hardware Accelerator!")
            
        except Exception as e:
            print(f"⚠️  [Recorder] NVENC gagal diinisialisasi ({type(e).__name__}). Fallback ke software x264.")
            
            # Buat ulang container dan stream bersih menggunakan CPU Encoder
            container.close()
            container = av.open(self.output_path, mode='w')
            
            stream = container.add_stream('libx264', rate=int(self.fps))
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = 'yuv420p'
            
            # Gunakan preset ultrafast agar beban CPU x264 sangat ringan
            stream.options = {'preset': 'ultrafast', 'tune': 'zerolatency'}

        # 2. Masuk ke Loop Utama Perekaman
        while not self.stopped or not self.frame_queue.empty():
            try:
                frame_bgr = self.frame_queue.get(timeout=0.01)
                
                # Ubah BGR (OpenCV) ke RGB untuk PyAV
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
                
                # Encode dan tulis ke disk
                for packet in stream.encode(av_frame):
                    container.mux(packet)
                    
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Amankan jika ada frame korup ditengah jalan agar thread tidak mati mendadak
                continue

        # Flush encoder buffer di akhir sesi
        try:
            for packet in stream.encode():
                container.mux(packet)
        except Exception:
            pass
            
        container.close()
        print("💾 [Recorder] Video berhasil disimpan dengan aman.")

# ==============================================================================
# THREAD 3: BACKGROUND AI PROCESSING THREAD
# ==============================================================================
class AIProcessingThread:
    def __init__(self, model_path, db_manager, matcher, res_opt):
        self.model = YOLO(model_path)
        self.db_manager = db_manager
        self.matcher = matcher
        self.res_opt = res_opt

        self.frame_idx = 0
        self.stopped = False

        # ---------------------------------------------------------------------------
        # FIX BOKS LAG: Ganti push-model (main loop overwrite slot tunggal) ke
        # queue-model dengan maxsize=2 (latest-frame queue).
        #
        # Problem lama: main loop push frame ke current_frame via lock. AI thread
        # ambil frame, proses, sementara main loop sudah overwrite 5-10x.
        # Karena AI thread set current_frame=None setelah ambil, ada jendela di mana
        # main loop push frame baru sebelum AI thread selesai proses yang lama →
        # AI thread langsung ambil frame "terlalu baru" relatif terhadap saat dia
        # publish boks ke shared_display_boxes → boks selalu 1-2 siklus di belakang
        # posisi aktual orang di frame yang sedang ditampilkan GUI.
        #
        # Solusi: maxsize=2 queue. Main loop put(frame, block=False) — jika queue
        # penuh (AI thread ketinggalan), drop frame lama dengan get() dulu baru put.
        # AI thread get() frame terbaru yang tersedia. Ini memastikan:
        # 1. AI thread SELALU proses frame terdekat dengan waktu GUI render
        # 2. Tidak ada frame.copy() sia-sia saat AI thread masih sibuk
        # 3. Lock contention hilang — queue sudah thread-safe built-in
        # ---------------------------------------------------------------------------
        self.frame_queue = queue.Queue(maxsize=2)

        # ---------------------------------------------------------------------------
        # YOLO GPU FP16: interval turun dari 2 → 1 (tiap frame AI).
        #
        # Sebelumnya interval=2 karena YOLO CPU ~15-25ms per call → bottleneck.
        # Sekarang YOLO jalan di GPU FP16 pada 640x360 input → ~3-6ms per call.
        # Budget 6ms jauh di bawah 1 AI cycle (~16ms di 60fps target).
        # Dengan interval=1: BoTSort dapat update posisi TIAP FRAME → Kalman filter
        # lebih stabil → track loss berkurang → TTFM lebih konsisten.
        # Side effect positif: eliminasi kebutuhan velocity interpolation bbox
        # karena posisi sudah selalu fresh dari YOLO real result, bukan stale.
        # ---------------------------------------------------------------------------
        self.yolo_interval = 1
        self.last_results = None

        # ==============================================================================
        # Fix 5: BoTSort PRE-WARM — Eliminasi cold-start latency ~50ms di frame pertama.
        #
        # Root cause: BoTSort tracker diinisialisasi lazy oleh Ultralytics di dalam
        # model.track() call pertama — mencakup alokasi Kalman filter state, ReID buffer,
        # dan beberapa CUDA op. Ini bisa 40-80ms di frame pertama yang kelihatan user.
        # Dengan dummy track call di init, BoTSort sudah "hangat" sebelum live dimulai.
        # Dummy frame ukuran identik (640x360) supaya Kalman filter init dengan
        # dimensi yang sama dengan live frames — tidak ada resize overhead.
        #
        # v2: Multi-pass warmup dengan noise image — warm YOLO NMS path juga.
        # Pass 1: blank frame (warm model load + CUDA context)
        # Pass 2: noise frame (warm NMS path yang aktif saat ada deteksi)
        # Pass 3: noise frame kecil untuk pastikan berbagai resolution path warm
        # ==============================================================================
        print("🔥 [BoTSort] Running BoTSort + YOLO tracker pre-warm pass (multi-pass)...")

        # Diagnosa CUDA availability untuk YOLO
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _gpu_name = _torch.cuda.get_device_name(0)
                _vram_total = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
                _vram_free  = (_torch.cuda.get_device_properties(0).total_memory -
                               _torch.cuda.memory_allocated(0)) / (1024**3)
                print(f"🎮 [GPU] {_gpu_name} | VRAM: {_vram_free:.1f}/{_vram_total:.1f} GB free")
            else:
                print(f"⚠️  [GPU] CUDA tidak tersedia untuk YOLO — akan jalan di CPU!")
        except Exception:
            pass

        _warmup_track_kwargs = dict(
            persist=True, tracker="botsort.yaml", verbose=False,
            classes=0, device='0', half=True, conf=0.18
        )
        try:
            # Pass 1: blank frame — warm CUDA context + model load
            _dummy_blank = np.zeros((360, 640, 3), dtype=np.uint8)
            _ = self.model.track(_dummy_blank, **_warmup_track_kwargs)

            # Pass 2: noise frame — warm NMS path (aktif saat ada banyak proposals)
            _dummy_noise = (np.random.rand(360, 640, 3) * 255).astype(np.uint8)
            _ = self.model.track(_dummy_noise, **_warmup_track_kwargs)

            # Pass 3: satu lagi noise frame untuk stabilkan Kalman filter init
            _dummy_noise2 = (np.random.rand(360, 640, 3) * 128).astype(np.uint8)
            _ = self.model.track(_dummy_noise2, **_warmup_track_kwargs)

            print("✅ [BoTSort] Pre-warm complete. Kalman filter state initialized.")
        except Exception as e:
            print(f"⚠️  [BoTSort] Pre-warm failed (non-critical): {e}")

        # Simpan last known bbox + velocity per track_id dalam koordinat 1920x1080.
        # Format: { track_id: { "bbox": (x1,y1,x2,y2), "vx": float, "vy": float,
        #                        "last_frame": int } }
        # velocity (vx, vy) adalah pixel per frame AI di centroid axis.
        # Dipakai untuk interpolasi prediktif di frame-frame yang di-skip YOLO
        # (sekarang interval=1 → praktis tidak dipakai, tapi dipertahankan sebagai
        # safety net kalau GPU bottleneck menyebabkan interval dinaikkan kembali).
        self._last_known_boxes = {}  # { track_id: { bbox, vx, vy, last_frame } }

    def start(self):
        t = threading.Thread(target=self.process_ai, args=())
        t.daemon = True
        t.start()
        return self

    def update_frame(self, frame_full, frame_yolo):
        # ---------------------------------------------------------------------------
        # Queue-based frame delivery: jika queue penuh (AI masih proses),
        # buang frame paling lama dan masukkan yang baru.
        # Ini garantikan AI thread selalu dapat frame se-fresh mungkin
        # tanpa blocking main loop sama sekali.
        # ---------------------------------------------------------------------------
        try:
            self.frame_queue.put_nowait((frame_full, frame_yolo))
        except queue.Full:
            try:
                self.frame_queue.get_nowait()  # buang frame stale
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait((frame_full, frame_yolo))
            except queue.Full:
                pass

    def process_ai(self):
        global shared_display_boxes, fps_inf_calc, attendance_final_log, \
               track_first_seen_frame, unknown_screenshot_buffer, verified_nrp_state, \
               track_first_seen_time

        # ==============================================================================
        # ADAPTIVE INFERENCE THROTTLE
        #
        # Dua mode yang saling eksklusif:
        #
        # MODE A — IDLE (semua track LOW / tidak ada track):
        #   Interval diambil dari res_opt.THROTTLE_IDLE_INTERVAL (25fps → 40ms/cycle).
        #   GPU punya ~75% idle time per cycle → clock sustained di mid-level.
        #   Tidak gosong, tidak throttle. Saat HIGH priority masuk, GPU bisa burst
        #   langsung dari sustained clock (bukan dari throttled clock).
        #
        #   Kenapa 25fps bukan 45fps?
        #   45fps idle → GPU duty cycle ~50%. Laptop GPU (terutama saat pakai MJPEG
        #   decode + ONNX face det concurrently) bisa mulai thermal throttle di sini.
        #   25fps idle → GPU duty cycle ~25%. Clock sangat stabil. Bedanya di TTFM:
        #   burst dari 25fps idle → GPU butuh ~0 ms untuk ramp up (sudah di sustained).
        #   burst dari throttled clock → GPU butuh 200-500ms untuk recover → TTFM
        #   pertama bisa 300-800ms lebih lambat dari seharusnya.
        #
        # MODE B — HIGH PRIORITY (ada track yang belum dikenali):
        #   TIDAK ADA SLEEP. Full throttle. TTFM race — setiap ms = TTFM naik.
        #   Window HIGH priority biasanya <3 detik per orang → thermal spike
        #   acceptable dan tidak sempat build up ke sustained throttle.
        #
        # Catatan desain penting:
        #   AI inference rate (25fps idle / unlimited HIGH) TERPISAH dari:
        #   - Camera capture rate (50fps — CameraThread, tidak terpengaruh)
        #   - GUI render rate (50fps — main loop, tidak terpengaruh)
        #   - Recording rate (50fps — VideoRecordingThread, tidak terpengaruh)
        #   BBox tracking memang bisa lagging 1-2 AI cycles (20-80ms) di GUI,
        #   tapi ini acceptable dan tidak kelihatan di video yang smooth 50fps.
        # ==============================================================================
        _THROTTLE_IDLE_INTERVAL_S = self.res_opt.THROTTLE_IDLE_INTERVAL  # 40ms (25fps)

        while not self.stopped:
            loop_start_time = time.time()

            # Ambil frame dari queue, block max 10ms
            try:
                frame, frame_yolo = self.frame_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            self.frame_idx += 1
            h_orig, w_orig = frame.shape[:2]
            # Scale factor dihitung sekali per frame, dipakai untuk semua bbox
            # Kamera 1280x720, YOLO input 640x360 → scale 2.0 untuk kedua axis
            scale_w = w_orig / 640.0
            scale_h = h_orig / 360.0

            # ------------------------------------------------------------------
            # YOLO + BoTSort tracking
            # Interval 1: YOLO GPU FP16 jalan TIAP FRAME (~3-6ms per call).
            # last_results tetap dipertahankan sebagai fallback kalau GPU spike,
            # tapi dalam kondisi normal run_yolo_this_frame selalu True.
            #
            # Fix 4: ADAPTIVE YOLO INTERVAL — kalau semua track sudah LOW priority
            # (semua orang sudah diabsen), naikkan interval ke 3 untuk hemat ~4ms/frame.
            # Kalau ada track HIGH priority (ada yang belum dikenali), tetap interval 1
            # untuk TTFM optimal. Logic ini dievaluasi tiap frame dengan O(n) check
            # di priority_registry yang paling banyak berisi 8-10 entry.
            # ------------------------------------------------------------------
            _has_high_priority = any(
                v["priority"] == "HIGH"
                for v in self.res_opt.priority_registry.values()
            )
            _effective_yolo_interval = 1 if _has_high_priority else 3

            t_yolo_start = time.time()
            run_yolo_this_frame = (self.frame_idx % _effective_yolo_interval == 0
                                   or self.frame_idx == 1
                                   or self.last_results is None)
            if run_yolo_this_frame:
                results = self.model.track(
                    frame_yolo,
                    persist=True,
                    tracker="botsort.yaml",
                    verbose=False,
                    classes=0,
                    device='0',   # GPU — dari CPU ke CUDA. FP16 potong VRAM ~50%
                    half=True,    # FP16 inference: latency 640x360 turun ~15ms → ~4ms
                    conf=0.18
                )
                self.last_results = results
            else:
                results = self.last_results
            self.res_opt.log_pipeline_time("yolo_time", (time.time() - t_yolo_start) * 1000)

            local_display_boxes = []

            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                # Kumpulkan crop untuk batched OSNet
                body_crop_batch  = []
                body_track_ids   = []
                body_centroids   = []
                body_nrp_bindings = []
                track_meta = []

                for box, track_id in zip(boxes, track_ids):
                    # ----------------------------------------------------------
                    # FIX BOKS LAG PART 2: Scale ke 1920x1080 DI SINI, bukan di GUI.
                    # AI thread yang tahu kapan koordinat ini valid (saat frame ini
                    # diproses). GUI hanya menggambar koordinat yang sudah final.
                    # ----------------------------------------------------------
                    x1 = int(box[0] * scale_w)
                    y1 = int(box[1] * scale_h)
                    x2 = int(box[2] * scale_w)
                    y2 = int(box[3] * scale_h)

                    centroid  = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    bbox_area = (x2 - x1) * (y2 - y1)

                    if (x2 - x1) < 20 or (y2 - y1) < 40:
                        continue

                    # Update last known position + velocity tiap YOLO frame
                    if run_yolo_this_frame:
                        if track_id in self._last_known_boxes:
                            prev = self._last_known_boxes[track_id]
                            frames_delta = self.frame_idx - prev["last_frame"]
                            if frames_delta > 0:
                                prev_cx = (prev["bbox"][0] + prev["bbox"][2]) / 2
                                prev_cy = (prev["bbox"][1] + prev["bbox"][3]) / 2
                                curr_cx = (x1 + x2) / 2
                                curr_cy = (y1 + y2) / 2
                                vx = (curr_cx - prev_cx) / frames_delta
                                vy = (curr_cy - prev_cy) / frames_delta
                            else:
                                vx = self._last_known_boxes[track_id]["vx"]
                                vy = self._last_known_boxes[track_id]["vy"]
                        else:
                            vx, vy = 0.0, 0.0
                        self._last_known_boxes[track_id] = {
                            "bbox": (x1, y1, x2, y2),
                            "vx": vx, "vy": vy,
                            "last_frame": self.frame_idx
                        }

                    # Crop di-assign DULU sebelum dipakai di blok manapun di bawah
                    crop_obj = frame[max(0, y1):min(h_orig, y2),
                                     max(0, x1):min(w_orig, x2)]

                    if track_id not in track_first_seen_frame:
                        track_first_seen_frame[track_id] = self.frame_idx
                        # FIX TTFM: catat wall-clock time untuk measurement akurat
                        track_first_seen_time[track_id] = time.time()
                        # Fix 3: pass frame_idx agar early bird mode bisa hitung frames_alive
                        self.res_opt.register_new_track(track_id, current_frame_idx=self.frame_idx)

                        # -------------------------------------------------------
                        # FIX STATE LOSS: Cek apakah track baru ini bisa di-match
                        # ke NRP yang sudah verified via body Re-ID.
                        # Ini handle kasus orang mendekat → BoTSort assign ID baru.
                        # -------------------------------------------------------
                        if crop_obj.size > 0:
                            quick_body_vec = self.matcher.extract_body_feature(crop_obj)
                            if quick_body_vec is not None:
                                _, candidate_nrp = self.matcher.match_body_reid(
                                    quick_body_vec, centroid,
                                    spatial_threshold=200, body_threshold=0.65,
                                    exclude_track_id=track_id
                                )
                                if candidate_nrp != "UNKNOWN" and candidate_nrp in verified_nrp_state:
                                    # Inherit state dari NRP yang sudah pernah verified
                                    self.res_opt.set_track_as_verified(track_id)
                                    self.res_opt.priority_registry[track_id]["nrp_name"] = candidate_nrp
                                    self.matcher.update_live_body(
                                        track_id, quick_body_vec,
                                        centroid=centroid, nrp=candidate_nrp
                                    )

                    run_face, run_body = self.res_opt.evaluate_gatekeepers(track_id, self.frame_idx)
                    # Baca status_absen DI SINI (setelah inheritance block di atas mungkin
                    # sudah set_track_as_verified), bukan sebelumnya.
                    status_absen    = self.res_opt.priority_registry[track_id]["status_absen"]

                    # FIX UNKNOWN STRANGER GHOST: jika NRP untuk track ini sudah ada di
                    # attendance_final_log (verified di sesi ini via track_id lain),
                    # langsung pakai nama itu tanpa menunggu face det lagi.
                    # Tanpa fix ini: track baru assign "SEARCHING...", tidak pernah
                    # trigger face match (karena NRP sudah di log), timeout 90 frame
                    # → UNKNOWN STRANGER padahal orangnya sudah diabsen.
                    nrp_from_registry = self.res_opt.priority_registry[track_id].get("nrp_name")
                    if not status_absen and nrp_from_registry and nrp_from_registry in attendance_final_log:
                        # Track ini punya nrp_name dari inheritance tapi status_absen belum di-set
                        # (edge case race condition) — force sync state
                        self.res_opt.set_track_as_verified(track_id)
                        status_absen = True

                    if status_absen:
                        identified_name = self.res_opt.priority_registry[track_id].get("nrp_name") or "VERIFIED"
                    else:
                        identified_name = "SEARCHING..."

                    track_meta.append({
                        "track_id": track_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "centroid": centroid, "bbox_area": bbox_area,
                        "crop_obj": crop_obj, "run_face": run_face, "run_body": run_body,
                        "status_absen": status_absen, "identified_name": identified_name
                    })

                    if run_body and crop_obj.size > 0:
                        body_crop_batch.append(crop_obj)
                        body_track_ids.append(track_id)
                        body_centroids.append(centroid)
                        nrp_to_bind = identified_name if status_absen else None
                        body_nrp_bindings.append(nrp_to_bind)

                # Batched OSNet — satu forward pass untuk semua track
                if body_crop_batch:
                    t_osnet_start = time.time()
                    body_vectors = self.matcher.extract_body_features_batch(body_crop_batch)
                    self.res_opt.log_pipeline_time("osnet_time", (time.time() - t_osnet_start) * 1000)
                    for i, vec in enumerate(body_vectors):
                        if vec is not None:
                            self.matcher.update_live_body(
                                body_track_ids[i], vec,
                                centroid=body_centroids[i],
                                nrp=body_nrp_bindings[i]
                            )

                # ==============================================================================
                # OPT v3: BATCHED FACE DETECTION + MATCHING PIPELINE
                #
                # Perubahan fundamental dari versi sebelumnya:
                #
                # LAMA: Loop serial per track:
                #   for each track:
                #     1. crop resize
                #     2. app_live.get(crop)        ← InsightFace det+rec (~38ms each)
                #     3. match_face_aligned()       ← USearch search serial
                #
                # BARU: Two-phase batched pipeline:
                #   Phase A - Face Detection (serial, tidak bisa di-batch di InsightFace):
                #     for each track: app_live.get(crop) → collect embeddings
                #   Phase B - Batch Matching (vectorized):
                #     matcher.batch_match_faces(all_embeddings) → satu matrix multiply
                #
                # Gain dari Phase B: eliminasi N serial USearch calls → 1 BLAS matmul.
                # Untuk 8 tracks: dari ~8×0.5ms → ~1ms total matching overhead.
                # Kecil tapi konsisten, plus Python overhead berkurang signifikan.
                #
                # PLUS: Adaptive threshold dari ResOptEngine. Setiap track bisa punya
                # threshold berbeda tergantung riwayat confidence-nya.
                # ==============================================================================

                # Phase A: Face detection per track, kumpulkan embeddings
                face_embeddings_for_batch = []   # embedding per track (atau None)
                face_det_meta = []               # metadata per track untuk Phase B

                for meta in track_meta:
                    track_id        = meta["track_id"]
                    crop_obj        = meta["crop_obj"]
                    run_face        = meta["run_face"]
                    centroid        = meta["centroid"]
                    status_absen    = meta["status_absen"]
                    identified_name = meta["identified_name"]
                    bbox_area       = meta["bbox_area"]
                    x1, y1, x2, y2 = meta["x1"], meta["y1"], meta["x2"], meta["y2"]

                    embedding_for_track = None
                    kps_for_track   = None
                    crop_for_face = None
                    face_crop_h = y2 - y1
                    face_crop_w = x2 - x1

                    # ======================================================
                    # VELOCITY-AWARE CROP BOOST
                    # Normalnya crop_h<40px → run_face=False total (terlalu kecil
                    # untuk reliable). Tapi kalau track ini bergerak MENUJU tengah
                    # frame (akan makin frontal sebentar lagi — kasus mahasiswa
                    # baru masuk dari tepi sambil berjalan menyamping), beri
                    # exception: tetap coba dengan upscale lebih agresif (4x)
                    # daripada full skip, supaya akumulasi EMA bisa mulai lebih
                    # awal alih-alih menunggu crop besar dulu.
                    # ======================================================
                    _velocity_boost_eligible = False
                    if run_face and crop_obj.size > 0 and face_crop_h < 40:
                        _vel_info = self._last_known_boxes.get(track_id)
                        if _vel_info is not None:
                            _velocity_boost_eligible = self.res_opt.is_moving_toward_center(
                                centroid[0], _vel_info.get("vx", 0.0), w_orig)

                    if crop_obj.size > 0 and run_face:
                        if face_crop_h < 40 or face_crop_w < 25:
                            if (_velocity_boost_eligible and
                                    face_crop_h >= self.res_opt._BOOST_MIN_CROP_H and
                                    face_crop_w >= 15):
                                # Exception: upscale agresif alih-alih skip total
                                crop_for_face = cv2.resize(
                                    crop_obj,
                                    (face_crop_w * self.res_opt._BOOST_UPSCALE_FACTOR,
                                     face_crop_h * self.res_opt._BOOST_UPSCALE_FACTOR),
                                    interpolation=cv2.INTER_CUBIC)
                            else:
                                run_face = False
                        elif face_crop_h < 60:
                            crop_for_face = cv2.resize(
                                crop_obj, (face_crop_w * 3, face_crop_h * 3),
                                interpolation=cv2.INTER_CUBIC)
                        elif face_crop_h < 120:
                            crop_for_face = cv2.resize(
                                crop_obj, (face_crop_w * 2, face_crop_h * 2),
                                interpolation=cv2.INTER_LINEAR)
                        else:
                            crop_for_face = crop_obj

                    if crop_obj.size > 0 and run_face and crop_for_face is not None:
                        t_fdet_start = time.time()
                        faces = self.db_manager.app_live.get(crop_for_face)
                        self.res_opt.log_pipeline_time("face_det_time",
                                                       (time.time() - t_fdet_start) * 1000)

                        if faces:
                            face = sorted(
                                faces,
                                key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]),
                                reverse=True
                            )[0]
                            embedding_for_track = face.embedding
                            kps_for_track = getattr(face, 'kps', None)
                    else:
                        run_face = False

                    face_embeddings_for_batch.append(embedding_for_track)
                    face_det_meta.append({
                        **meta,
                        "run_face": run_face,
                        "face_crop_h": face_crop_h,
                        "embedding": embedding_for_track,
                        "kps": kps_for_track,
                        "identified_name": identified_name,
                    })

                # Phase B: Batch matching — satu vectorized call untuk semua tracks
                t_frec_start = time.time()
                batch_match_results = self.matcher.batch_match_faces(
                    face_embeddings_for_batch, threshold=0.65)

                # EMA SUPPORT: raw best-candidate + similarity TANPA threshold cutoff.
                # Dipakai untuk dua hal sekaligus (satu matmul, reused):
                #   1. Adaptive threshold soft-retry (gantikan bug lama yang selalu
                #      pakai _raw_sim=0.0 saat UNKNOWN — sekarang pakai nilai asli)
                #   2. EMA accumulation untuk kasus mahasiswa berjalan (Jalur 2)
                batch_match_raw = self.matcher.batch_match_faces_raw(
                    face_embeddings_for_batch)

                self.res_opt.log_pipeline_time("face_rec_time",
                                               (time.time() - t_frec_start) * 1000)

                # Phase C: Process match results per track
                for i, meta in enumerate(face_det_meta):
                    track_id        = meta["track_id"]
                    centroid        = meta["centroid"]
                    status_absen    = meta["status_absen"]
                    identified_name = meta["identified_name"]
                    bbox_area       = meta["bbox_area"]
                    x1, y1, x2, y2 = meta["x1"], meta["y1"], meta["x2"], meta["y2"]
                    run_face        = meta["run_face"]
                    face_crop_h     = meta["face_crop_h"]
                    raw_embedding   = meta["embedding"]
                    raw_kps         = meta["kps"]

                    matched_nrp, face_similarity = batch_match_results[i]
                    raw_candidate_nrp, raw_sim    = batch_match_raw[i]

                    # ==========================================================
                    # JALUR 1 — DIRECT HIT sudah dicek lewat batch_match_results
                    # (matched_nrp != "UNKNOWN" berarti raw_sim >= 0.65 di SATU
                    # frame ini). Tidak ada delay — TTFM untuk kasus mudah tetap
                    # cepat seperti sebelumnya.
                    # ==========================================================

                    # OPT v3: Adaptive threshold check — sekarang pakai raw_sim
                    # ASLI (bukan selalu 0.0), supaya soft-threshold retry punya
                    # sinyal yang benar untuk decide kapan melunakkan threshold.
                    if run_face and raw_embedding is not None:
                        if matched_nrp == "UNKNOWN":
                            adaptive_threshold = self.res_opt.get_adaptive_threshold(
                                track_id, raw_sim, self.frame_idx, base_threshold=0.65)
                            if adaptive_threshold < 0.65:
                                retry_nrp, retry_sim = self.matcher.match_face(
                                    raw_embedding, threshold=adaptive_threshold)
                                if retry_nrp != "UNKNOWN":
                                    matched_nrp = retry_nrp
                                    face_similarity = retry_sim
                        else:
                            self.res_opt.get_adaptive_threshold(
                                track_id, raw_sim, self.frame_idx, base_threshold=0.65)

                    # ==========================================================
                    # JALUR 2 — EMA ACCUMULATION (khusus mahasiswa berjalan)
                    #
                    # Hanya dipanggil kalau Jalur 1 (direct hit) DAN adaptive
                    # soft-retry di atas SAMA-SAMA gagal. Tujuannya menangkap
                    # kasus confidence noisy yang individual tidak pernah cukup
                    # tinggi, tapi rata-ratanya (setelah 4-5 akumulasi) solid.
                    #
                    # raw_candidate_nrp tetap diberikan ke EMA walau similarity-nya
                    # rendah — ResOptEngine yang decide kapan EMA matang & cukup
                    # tinggi untuk declare match (lihat accumulate_face_ema).
                    #
                    # pose_weight: dihitung dari 5 keypoints InsightFace (kps).
                    # Wajah frontal → weight≈1.0 (kontribusi penuh ke EMA).
                    # Wajah profil/3-4 (kasus berjalan menyamping) → weight turun
                    # → sample tetap masuk (sample_count tetap nambah) tapi
                    # pengaruhnya ke ema_score dikecilkan, supaya pose buruk
                    # tidak menyeret EMA turun secara tidak proporsional.
                    # ==========================================================
                    if matched_nrp == "UNKNOWN" and run_face and raw_embedding is not None:
                        pose_weight = self.res_opt.estimate_pose_weight(raw_kps)
                        ema_match_nrp, ema_score = self.res_opt.accumulate_face_ema(
                            track_id, raw_candidate_nrp, raw_sim, pose_weight=pose_weight)
                        if ema_match_nrp is not None:
                            matched_nrp = ema_match_nrp
                            face_similarity = ema_score

                    if matched_nrp != "UNKNOWN" and matched_nrp is not None:
                        if matched_nrp not in attendance_final_log:
                            total_frames_elapsed = self.frame_idx - track_first_seen_frame[track_id] + 1
                            ttfm_milidetik = (time.time() - track_first_seen_time.get(track_id, time.time())) * 1000
                            attendance_final_log[matched_nrp] = {
                                "detected_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "ttfm_frames": total_frames_elapsed,
                                "ttfm_ms": ttfm_milidetik,
                                "confidence": face_similarity,
                                "pipeline_latencies_ms": None
                            }
                        self.res_opt.set_track_as_verified(track_id)
                        self.res_opt.priority_registry[track_id]["nrp_name"] = matched_nrp
                        identified_name = matched_nrp
                        verified_nrp_state[matched_nrp] = True

                    elif run_face and not status_absen and self.frame_idx - track_first_seen_frame[track_id] > 8:
                        _crop_zone_weight = 0.4 if face_crop_h < 120 else 1.0
                        _current_stranger = self.res_opt.priority_registry[track_id].get("stranger_frame_count", 0)
                        self.res_opt.priority_registry[track_id]["stranger_frame_count"] = (
                            _current_stranger + _crop_zone_weight
                        )

                        real_body_vector = self.matcher.live_body_registry.get(
                            track_id, {}).get("body_vectors", [])
                        if real_body_vector:
                            current_vec = real_body_vector[-1]
                            fallback_track_id, fallback_nrp = self.matcher.match_body_reid(
                                current_vec, centroid,
                                spatial_threshold=150, body_threshold=0.70,
                                exclude_track_id=track_id
                            )
                            if fallback_nrp != "UNKNOWN":
                                if fallback_nrp not in attendance_final_log:
                                    total_frames_elapsed = self.frame_idx - track_first_seen_frame[track_id] + 1
                                    ttfm_milidetik = (time.time() - track_first_seen_time.get(track_id, time.time())) * 1000
                                    attendance_final_log[fallback_nrp] = {
                                        "detected_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "ttfm_frames": total_frames_elapsed,
                                        "ttfm_ms": ttfm_milidetik,
                                        "confidence": 0.0,
                                        "pipeline_latencies_ms": None
                                    }
                                self.res_opt.set_track_as_verified(track_id)
                                self.res_opt.priority_registry[track_id]["nrp_name"] = fallback_nrp
                                identified_name = fallback_nrp
                                verified_nrp_state[fallback_nrp] = True

                    # Safety net + stranger labeling per track (dari face_det_meta)
                    _nrp_check = self.res_opt.priority_registry[track_id].get("nrp_name")
                    _already_known = (_nrp_check and _nrp_check in verified_nrp_state)
                    _stranger_count = self.res_opt.priority_registry[track_id].get("stranger_frame_count", 0)
                    _crop_from_meta = meta.get("crop_obj")
                    crop_obj = _crop_from_meta if _crop_from_meta is not None else frame[max(0,y1):min(h_orig,y2), max(0,x1):min(w_orig,x2)]

                    if identified_name == "SEARCHING..." and not _already_known and \
                            _stranger_count > 30:
                        identified_name = "UNKNOWN STRANGER"
                        if track_id not in unknown_screenshot_buffer:
                            unknown_screenshot_buffer[track_id] = []
                        if len(unknown_screenshot_buffer[track_id]) < 3 and crop_obj.size > 0:
                            unknown_screenshot_buffer[track_id].append((bbox_area, frame.copy()))
                            unknown_screenshot_buffer[track_id] = sorted(
                                unknown_screenshot_buffer[track_id],
                                key=lambda x: x[0], reverse=True
                            )

                    local_display_boxes.append({
                        "bbox": (x1, y1, x2, y2),
                        "track_id": track_id,
                        "name": identified_name,
                        "status_absen": status_absen
                    })

            # ------------------------------------------------------------------
            # Bersihkan track yang sudah tidak ada di frame ini dari last_known_boxes
            # supaya dict tidak tumbuh tak terbatas
            # ------------------------------------------------------------------
            if run_yolo_this_frame and results and results[0].boxes and results[0].boxes.id is not None:
                active_ids = set(results[0].boxes.id.cpu().numpy().astype(int))
                stale_ids  = [tid for tid in self._last_known_boxes if tid not in active_ids]
                for tid in stale_ids:
                    self._last_known_boxes.pop(tid, None)

            with shared_boxes_lock:
                shared_display_boxes = local_display_boxes

            fps_inf_calc = 1.0 / (time.time() - loop_start_time + 1e-6)

            # ==============================================================================
            # ADAPTIVE THROTTLE EXECUTION
            # Cek kondisi throttle SETELAH semua inference selesai di cycle ini.
            # MODE IDLE (25fps): aktif kalau semua track LOW priority (semua sudah absen)
            # ATAU tidak ada track sama sekali. GUI tetap 50fps dari CameraThread.
            # MODE HIGH PRIORITY: tidak ada sleep → full blast untuk minimasi TTFM.
            # ==============================================================================
            _cycle_elapsed = time.time() - loop_start_time
            _should_throttle = not any(
                v["priority"] == "HIGH"
                for v in self.res_opt.priority_registry.values()
            )
            if _should_throttle:
                _sleep_needed = _THROTTLE_IDLE_INTERVAL_S - _cycle_elapsed
                if _sleep_needed > 0.001:  # minimum 1ms threshold untuk hindari overhead sleep
                    time.sleep(_sleep_needed)


# ==============================================================================
# PHASE 1: INITIALIZATION
# ==============================================================================
print("🚀 [Step 1] Booting up Face DB Manager...")
db_manager = FaceDBManager(base_dir=FACE_DB_DIR, target_frames=25, ctx_id=0, det_size=(640, 640))
db_manager.extract_and_register_class()

face_db_file = os.path.join(FACE_DB_DIR, "registered_class_vectors.pkl")
matcher  = IdentityMatcher(face_db_path=face_db_file, body_history_size=10)
res_opt  = ResOptEngine(fps_target=50)

print("\n📸 [Hardware] Launching Non-Blocking Camera Thread (Isolated I/O)...")
cam_thread = CameraCaptureThread(src=0).start()

print("🧠 [Engine] Launching Isolated AI Core Background Thread...")
ai_engine = AIProcessingThread(YOLO_MODEL_PATH, db_manager, matcher, res_opt).start()

print("📹 [Disk I/O] Launching Asynchronous Hardware Video Recording Thread...")
video_recorder = HardwareVideoRecorder(OUTPUT_VIDEO_PATH, fps=50.0, resolution=(1280, 720)).start()

print("\n🎬 [Engine] Quad-Threaded Zero-Blocking Engine is Live. Press 'Q' to exit.")
print("-" * 80)

# ==============================================================================
# THREAD UTAMA: GRAPHICS RENDERING & GUI
# Tidak ada lagi scaling koordinat di sini — semua bbox sudah dalam 1280x720
# ==============================================================================
# GUI loop: feed langsung dari CameraThread (selalu fresh, tidak nunggu AI),
# bbox dari shared_display_boxes (koordinat sudah 1280x720, AI thread yang
# scale, tidak perlu scaling ulang di sini). FPS overlay dimatikan — tidak
# ada lagi cv2.putText untuk angka FPS, supaya tampilan bersih untuk demo.
# fps_inf_calc tetap dihitung di process_ai() untuk laporan akhir di terminal,
# hanya tidak ditampilkan di overlay GUI.
# ==============================================================================
# _last_rendered_frame_id: track frame mana yang terakhir kali dirender dan
# ditulis ke video. GUI loop spinning lebih cepat dari 50fps (tidak ada sleep),
# jadi tanpa dedup ini setiap frame kamera baru akan dirender 3-8x sebelum
# frame kamera berikutnya datang — menghasilkan video yang isinya 80% frame
# duplikat, motion terlihat stuttery/25fps meski metadata bilang 50fps.
_last_rendered_frame_id = -1
last_render_time = time.perf_counter()
frame_count = 0
last_fps_check = time.time()

latest_active_frame = None

while True:
    current_time = time.perf_counter()
    
    ret, frame, frame_id = cam_thread.read_latest_frame()
    if ret and frame is not None:
        latest_active_frame = frame
        
        if frame_id != _last_rendered_frame_id:
            _last_rendered_frame_id = frame_id
            # Resize untuk YOLO — tetap 640x360
            frame_yolo_light = cv2.resize(frame, (640, 360))
            ai_engine.update_frame(frame, frame_yolo_light)

    if latest_active_frame is None:
        time.sleep(0.001)
        continue

    # Jalankan render persis setiap batas interval 60 FPS (16.67ms)
    if current_time - last_render_time >= (1.0 / 50.0):
        last_render_time = current_time

        # Gunakan frame yang tersedia untuk digambar ulang bersama kotak pelacak baru
        display_frame = latest_active_frame.copy()

        with shared_boxes_lock:
            active_boxes = list(shared_display_boxes)

        for obj in active_boxes:
            # Koordinat langsung dipakai — sudah 1280x720, tidak perlu scale
            x1, y1, x2, y2  = obj["bbox"]
            track_id         = obj["track_id"]
            identified_name  = obj["name"]
            status_absen     = obj["status_absen"]

            box_color = (0, 255, 0) if status_absen else (0, 165, 255)
            if identified_name == "UNKNOWN STRANGER":
                box_color = (0, 0, 255)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(display_frame, f"ID:{track_id} | {identified_name}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        cv2.imshow("FaceSORT Live System Engine - 1280x720", display_frame)
        video_recorder.write_frame_async(display_frame)

    else:
        time.sleep(0.001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Teardown
ai_engine.stopped = True
video_recorder.stopped = True
cam_thread.stop()
cv2.destroyAllWindows()

# ==============================================================================
# PHASE 3 & 4: POST PROCESSING & GRANULAR REPORTING
# ==============================================================================
print("\n📸 [Post-Processing] Archiving anomaly snapshots...")
for track_id, shot_list in unknown_screenshot_buffer.items():
    for rank_idx, (area, img_data) in enumerate(shot_list):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Track_{track_id}_Rank_{rank_idx+1}_{timestamp_str}.jpg"
        cv2.imwrite(os.path.join(ANOMALY_DIR, filename), img_data)

# ==============================================================================
# FIX BUG LATENCY: Isi pipeline_latencies_ms di sini, BUKAN saat entry pertama
# dibuat di AI thread.
#
# Root cause: get_average_pipeline_latencies() dipanggil di frame match pertama,
# saat itu osnet_time dan face_rec_time bisa masih list kosong (belum pernah
# ter-trigger di frame tersebut) → return 0.0 ms yang menyesatkan.
#
# Dengan mengambil snapshot di teardown:
# 1. Semua komponen sudah ter-log ratusan kali sepanjang session → average akurat.
# 2. OSNet pasti sudah jalan (warmup + batch inference) → tidak pernah 0.0 ms lagi.
# 3. USearch pasti sudah jalan minimal 1x per mahasiswa yang ter-log.
#
# Entry yang sudah ada pipeline_latencies_ms (bukan None) tidak di-overwrite,
# sebagai defensive guard kalau ada path lain di masa depan yang isi duluan.
# ==============================================================================
final_session_latencies = res_opt.get_average_pipeline_latencies()
for nrp in attendance_final_log:
    if attendance_final_log[nrp]["pipeline_latencies_ms"] is None:
        attendance_final_log[nrp]["pipeline_latencies_ms"] = final_session_latencies

print("\n" + "="*90)
print("📊 LAPORAN GRANULAR INDIVIDUAL MAHASISWA BERHASIL TERABSEN")
print("="*90)
for nrp, metrics in attendance_final_log.items():
    print(f"🎓 [Mahasiswa ID: {nrp}]")
    print(f"     - Kunci Kehadiran Pertama : {metrics['ttfm_frames']} Frame")
    print(f"     - Kinerja Respon TTFM     : {metrics['ttfm_ms']:.2f} ms")
    print(f"     - Confidence Vector       : {metrics['confidence']:.4f}")
    print(f"     - Profil Latensi Komputasi Akhir (Rata-Rata Session):")
    lat = metrics['pipeline_latencies_ms'] or {}
    # FIX BUG FORMAT: Ubah dari :.3f ke :.4f agar USearch sub-ms tidak truncate ke 0.000
    # USearch untuk 549 vektor biasanya 0.05-0.3ms — butuh 4 desimal untuk kelihatan.
    print(f"          * Deteksi Person (YOLOv8 640p Downscale CPU)      : {lat.get('yolo_time', 0.0):.3f} ms")
    print(f"          * Deteksi Wajah (InsightFace det+rec only 320x320): {lat.get('face_det_time', 0.0):.3f} ms")
    print(f"          * Matcher USearch (ArcFace Cosine)                : {lat.get('face_rec_time', 0.0):.4f} ms")
    print(f"          * Fitur Baju (OSNet AIN GPU)                      : {lat.get('osnet_time', 0.0):.3f} ms")
    print("-" * 90)