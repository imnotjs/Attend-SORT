import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# ==============================================================================
# ARCFACE ALIGNMENT VIA det_10g NATIVE 5-POINT KPS
# ==============================================================================
_ARCFACE_112_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face_arcface(img_bgr: np.ndarray, kps: np.ndarray,
                       output_size: int = 112) -> np.ndarray:
    src = kps.astype(np.float32)
    dst = _ARCFACE_112_TEMPLATE.copy()
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None
    aligned = cv2.warpAffine(img_bgr, M, (output_size, output_size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return aligned


class _LightFaceEngine:
    """
    Engine minimal: hanya load det_10g (RetinaFace) + w600k_r50 (ArcFace recognition).
    Skip landmark models (1k3d68, 2d106det) dan genderage.

    OPT v3 [COLD-START FIX]:
    - Warmup menggunakan MULTIPLE REAL FACE IMAGE sebagai ganti synthetic face.
    - Real faces dari disk jauh lebih reliabel memicu full detection path (NMS + crop
      + align + recognition) karena det_10g ditraining pada distribusi wajah nyata,
      bukan wajah sintetis sederhana.
    - Jika tidak ada real face tersedia, fallback ke multi-pass synthetic warmup
      dengan variasi posisi/rotasi untuk tingkatkan probabilitas detection fire.
    - Tambah: direct recognition model warmup via get_feat() untuk garantikan
      buffer ONNX recognition ter-alokasikan terlepas dari detection hasil.
    """
    def __init__(self, ctx_id=0, det_size=(320, 320),
                 providers=('CUDAExecutionProvider', 'CPUExecutionProvider'),
                 warmup_image_paths=None):
        self.det_size = det_size
        self._app = FaceAnalysis(
            name='buffalo_l',
            allowed_modules=['detection', 'recognition'],
            providers=list(providers)
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)
        self._warmup_image_paths = warmup_image_paths or []

    def get(self, img):
        return self._app.get(img)

    def run_warmup(self):
        """
        Multi-strategy warmup untuk eliminasi cold-start latency ~200ms+ di frame pertama.

        Strategy 1 (TERBAIK): Feed real face image dari database jika tersedia.
          det_10g reliably detect wajah nyata → full path ter-warm (NMS + crop +
          align + w600k_r50 forward pass). Ini yang paling efektif.

        Strategy 2: Multi-pass synthetic face dengan variasi rotasi/posisi.
          Beberapa variasi meningkatkan probabilitas det_10g fire. Jika salah satu
          terdeteksi → full path ter-warm.

        Strategy 3 (FALLBACK): Direct recognition warmup via get_feat() internal.
          Bypass detection, langsung warm ONNX buffer recognition model.
          Tidak sebaik full path tapi garantikan recognition buffer ter-alokasikan.

        Returns: str status warmup ("full_real", "full_synthetic", "partial_rec_only")
        """
        warmup_status = "not_started"

        # --- Strategy 1: Real face images ---
        if self._warmup_image_paths:
            for img_path in self._warmup_image_paths[:3]:  # max 3 gambar
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    # Resize ke det_size-compatible input (lebih kecil = lebih cepat warmup)
                    img_resized = cv2.resize(img, (320, 320))
                    faces = self._app.get(img_resized)
                    if faces:
                        warmup_status = "full_real"
                        # Satu detection sudah cukup untuk warm full path
                        break
                except Exception:
                    continue

        if warmup_status == "full_real":
            # Lakukan satu pass lagi untuk warm buffer pada ukuran det yang berbeda
            try:
                if self._warmup_image_paths:
                    img = cv2.imread(self._warmup_image_paths[0])
                    if img is not None:
                        img_resized = cv2.resize(img, self.det_size)
                        self._app.get(img_resized)
            except Exception:
                pass
            return warmup_status

        # --- Strategy 2: Multi-pass synthetic face dengan variasi ---
        synthetic_variants = []

        # Variant A: Frontal, normal lighting simulation
        _img_a = np.full((320, 320, 3), [175, 145, 115], dtype=np.uint8)
        cv2.ellipse(_img_a, (160, 155), (72, 95), 0, 0, 360, (205, 175, 145), -1)
        cv2.circle(_img_a, (128, 128), 13, (35, 25, 20), -1)
        cv2.circle(_img_a, (192, 128), 13, (35, 25, 20), -1)
        cv2.circle(_img_a, (128, 128), 6, (210, 210, 210), -1)
        cv2.circle(_img_a, (192, 128), 6, (210, 210, 210), -1)
        _nose_a = np.array([[160, 152], [147, 183], [173, 183]], np.int32)
        cv2.fillPoly(_img_a, [_nose_a], (115, 85, 70))
        cv2.ellipse(_img_a, (160, 208), (30, 13), 0, 0, 180, (75, 45, 40), -1)
        _img_a = cv2.GaussianBlur(_img_a, (3, 3), 0)
        synthetic_variants.append(_img_a)

        # Variant B: Slightly smaller / farther face (simulates person at distance)
        _img_b = np.full((320, 320, 3), [160, 135, 110], dtype=np.uint8)
        cv2.ellipse(_img_b, (160, 160), (55, 72), 0, 0, 360, (190, 160, 130), -1)
        cv2.circle(_img_b, (138, 140), 10, (35, 25, 20), -1)
        cv2.circle(_img_b, (182, 140), 10, (35, 25, 20), -1)
        _nose_b = np.array([[160, 155], [151, 177], [169, 177]], np.int32)
        cv2.fillPoly(_img_b, [_nose_b], (115, 85, 70))
        cv2.ellipse(_img_b, (160, 200), (24, 10), 0, 0, 180, (75, 45, 40), -1)
        synthetic_variants.append(cv2.GaussianBlur(_img_b, (3, 3), 0))

        # Variant C: Slightly tilted — memaksa warp path berbeda
        _img_c = _img_a.copy()
        M_rot = cv2.getRotationMatrix2D((160, 160), 8, 1.0)
        _img_c = cv2.warpAffine(_img_c, M_rot, (320, 320))
        synthetic_variants.append(_img_c)

        for variant in synthetic_variants:
            try:
                faces = self._app.get(variant)
                if faces:
                    warmup_status = "full_synthetic"
                    break
            except Exception:
                continue

        if warmup_status != "full_synthetic":
            # --- Strategy 3: Direct recognition model warmup ---
            try:
                # Buat aligned crop 112x112 — ukuran input w600k_r50
                _rec_img = np.full((112, 112, 3), [200, 170, 140], dtype=np.uint8)
                # Add minimal texture agar lebih dekat distribusi training
                cv2.circle(_rec_img, (40, 45), 8, (30, 20, 15), -1)
                cv2.circle(_rec_img, (72, 45), 8, (30, 20, 15), -1)
                cv2.ellipse(_rec_img, (56, 80), (18, 8), 0, 0, 180, (60, 35, 30), -1)
                _rec_img = cv2.GaussianBlur(_rec_img, (3, 3), 0)

                # Warm recognition model via get_feat() (InsightFace internal API)
                for _model in self._app.models.values():
                    if hasattr(_model, 'get_feat'):
                        # get_feat() expect list of aligned images
                        _model.get_feat([_rec_img])
                        warmup_status = "partial_rec_only"
                        break
                    elif hasattr(_model, 'forward'):
                        # Some versions expose forward directly
                        try:
                            import torch
                            _inp = torch.from_numpy(
                                _rec_img.transpose(2, 0, 1).astype(np.float32)[None]
                            )
                            if next(_model.parameters(), None) is not None:
                                _inp = _inp.to(next(_model.parameters()).device)
                            with torch.no_grad():
                                _model(_inp)
                            warmup_status = "partial_rec_only"
                        except Exception:
                            pass
                        break
            except Exception:
                warmup_status = "failed"

        return warmup_status


class FaceDBManager:
    def __init__(self, base_dir, target_frames=25, ctx_id=0, det_size=(640, 640)):
        self.base_dir = base_dir
        self.db_path = os.path.join(base_dir, "registered_class_vectors.pkl")
        self.log_path = os.path.join(base_dir, "selected_frames_log.txt")
        self.target_frames = target_frames
        self.database_registry = {}

        print("🧠 [Engine] Initializing InsightFace Buffalo_L Core (Registration Engine)...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # ==============================================================================
        # OPT v3: Kumpulkan sample face images dari database untuk dipakai warmup.
        # Ini kunci Fix Cold-Start — real faces jauh lebih reliabel untuk warm det path.
        # Ambil satu gambar dari max 2 student pertama (tidak perlu banyak).
        # ==============================================================================
        _warmup_face_paths = self._collect_warmup_face_samples(max_samples=3)

        print("🧠 [Engine] Initializing Lightweight Live Engine (det+rec only, 320x320)...")
        self.app_live = _LightFaceEngine(
            ctx_id=ctx_id,
            det_size=(320, 320),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            warmup_image_paths=_warmup_face_paths
        )
        print("✅ [Engine] Live engine ready — landmark models excluded from live path.")

        # ==============================================================================
        # OPT v3: MULTI-STRATEGY WARMUP
        # Warmup sekarang dijalankan dari _LightFaceEngine.run_warmup() yang memiliki
        # fallback bertingkat: real face → multi-pass synthetic → direct rec warmup.
        # ==============================================================================
        print("🔥 [Engine] Running LightFaceEngine warmup (multi-strategy)...")
        _status = self.app_live.run_warmup()
        _status_labels = {
            "full_real": "✅ FULL PIPELINE (real face detected) — cold-start eliminated",
            "full_synthetic": "✅ FULL PIPELINE (synthetic face detected) — cold-start eliminated",
            "partial_rec_only": "⚠️  PARTIAL (recognition only) — 1st-frame det may be slightly slow",
            "failed": "❌ WARMUP FAILED — cold-start latency will occur",
            "not_started": "⚠️  WARMUP SKIPPED",
        }
        print(f"   → Warmup Status: {_status_labels.get(_status, _status)}")

        # ==============================================================================
        # OPT v3: SECOND PASS WARMUP — eliminasi sisa buffer alokasi ONNX.
        # Beberapa ONNX operator lazy-allocate pada pass pertama tapi masih ada
        # alokasi minor di pass ke-2 sebelum fully steady-state.
        # Dua kali warmup memastikan semua path sudah steady sebelum live stream.
        # ==============================================================================
        if _status in ("full_real", "full_synthetic"):
            print("🔥 [Engine] Running second warmup pass (steady-state ONNX buffer lock-in)...")
            self.app_live.run_warmup()
            print("✅ [Engine] Second pass complete. ONNX in fully steady-state.")

        self._load_existing_db()

    def _collect_warmup_face_samples(self, max_samples=3):
        """
        Kumpulkan path gambar wajah nyata dari database folder untuk dipakai warmup.
        Prioritaskan gambar 'frame_000001.jpg' (frame pertama tiap student = biasanya frontal).
        """
        samples = []
        if not os.path.exists(self.base_dir):
            return samples

        student_folders = sorted([
            f for f in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, f))
        ])

        for student_id in student_folders:
            if len(samples) >= max_samples:
                break
            student_path = os.path.join(self.base_dir, student_id)
            # Cari frame terkecil (cepat load) yang ada di folder
            for candidate in ["frame_000001.jpg", "frame_000002.jpg", "frame_000003.jpg"]:
                full_path = os.path.join(student_path, candidate)
                if os.path.exists(full_path):
                    samples.append(full_path)
                    break
            else:
                # Fallback: cari file frame_ pertama apapun
                frame_files = sorted([
                    f for f in os.listdir(student_path)
                    if f.startswith("frame_") and f.endswith(".jpg")
                ])
                if frame_files:
                    samples.append(os.path.join(student_path, frame_files[0]))

        return samples

    def _load_existing_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.database_registry = pickle.load(f)
                print(f"💾 [Database] Existing registry loaded. Found {len(self.database_registry)} registered students.")
            except Exception as e:
                print(f"⚠️ [Warning] Failed to load existing database, creating a fresh one. Error: {e}")
                self.database_registry = {}
        else:
            print("📁 [Database] No existing database found. Creating a fresh registry.")
            self.database_registry = {}

    def extract_and_register_class(self):
        student_folders = [f for f in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, f))]
        new_students = [s for s in student_folders if s not in self.database_registry]

        if not new_students:
            print("✅ [Database] All student folders are up-to-date. No new registration needed.")
            return

        print(f"👥 [Database] Detected {len(new_students)} new student IDs for incremental registration.")
        log_records = []

        for student_id in sorted(new_students):
            student_path = os.path.join(self.base_dir, student_id)
            print(f"\n📂 [Processing] Compiling face cluster for ID: {student_id}...")

            frame_files = sorted([f for f in os.listdir(student_path) if f.startswith("frame_") and f.endswith(".jpg")])
            if not frame_files:
                print(f"⚠️ [Warning] Folder {student_id} has no valid 'frame_xxxxxx.jpg'. Skipped.")
                continue

            all_valid_faces = []

            for frame_file in frame_files:
                img_path = os.path.join(student_path, frame_file)
                img = cv2.imread(img_path)
                if img is None: continue

                faces = self.app.get(img)
                if not faces: continue

                face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                landmarks = face.kps
                if landmarks is None:
                    continue

                embedding = face.embedding

                if embedding is not None:
                    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
                    dist_left = np.linalg.norm(left_eye - nose)
                    dist_right = np.linalg.norm(right_eye - nose)
                    symmetry_score = abs(dist_left - dist_right)

                    all_valid_faces.append({
                        'filename': frame_file,
                        'embedding': embedding,
                        'landmarks': landmarks,
                        'symmetry': symmetry_score
                    })

            if not all_valid_faces:
                print(f"❌ [Error] Zero valid faces detected in folder {student_id}.")
                continue

            all_valid_faces = sorted(all_valid_faces, key=lambda x: x['symmetry'])
            anchor_face = all_valid_faces[0]

            selected_faces = [anchor_face]
            anchor_landmark = anchor_face['landmarks']

            remaining_faces = all_valid_faces[1:]
            if len(remaining_faces) > (self.target_frames - 1):
                for f in remaining_faces:
                    f['deviation'] = np.mean(np.linalg.norm(f['landmarks'] - anchor_landmark, axis=1))

                remaining_faces = sorted(remaining_faces, key=lambda x: x['deviation'], reverse=True)
                selected_faces.extend(remaining_faces[:self.target_frames - 1])
            else:
                selected_faces.extend(remaining_faces)

            final_embeddings = np.array([f['embedding'] for f in selected_faces], dtype=np.float32)
            final_filenames = [f['filename'] for f in selected_faces]

            self.database_registry[student_id] = {
                'embeddings': final_embeddings,
                'selected_files': final_filenames
            }

            log_records.append(f"=== ID: {student_id} ===")
            log_records.append(f"Anchor Utama Frontal: {final_filenames[0]}")
            log_records.append(f"Selected Distribution Cluster: {', '.join(final_filenames)}")
            log_records.append("-" * 60)

            n_total_valid = len(all_valid_faces)
            n_registered  = len(final_filenames)
            warn = " ⚠️  LOW COVERAGE" if n_registered < 10 else ""
            print(f"✨ [Success] ID {student_id}: {n_registered}/{n_total_valid} vectors registered (cap={self.target_frames}){warn}")

        self._save_db(log_records)

    def _save_db(self, log_records):
        print("\n💾 Committing changes to vector database persistence file...")
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.database_registry, f)
        print(f"✅ Database update complete -> {self.db_path}")

        if log_records:
            with open(self.log_path, 'a') as f:
                f.write("\n" + "\n".join(log_records))
            print(f"📝 Registry logs updated -> {self.log_path}")


if __name__ == "__main__":
    PROJECT_BASE_DIR = r"C:\Projects\FaceSORT Live Demo\Face Database"
    db_manager = FaceDBManager(base_dir=PROJECT_BASE_DIR, target_frames=25, ctx_id=0, det_size=(640, 640))
    db_manager.extract_and_register_class()