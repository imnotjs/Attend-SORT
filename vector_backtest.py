import os
import cv2
import numpy as np
import pickle
import time
import random
from ultralytics import YOLO
from usearch.index import Index
from face_db_manager import _LightFaceEngine

# ==============================================================================
# KONFIGURASI
# ==============================================================================
BASE_DIR = r"C:\Projects\FaceSORT Live Demo"
VECTOR_DATABASE_FILE = os.path.join(BASE_DIR, "Face Database", "registered_class_vectors.pkl")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "widerperson_best.pt")

NUM_ITERATIONS        = 1000
SIMILARITY_THRESHOLD  = 0.65   # Threshold live system

# Mahasiswa dianggap "perlu perhatian" jika salah satu kondisi ini terpenuhi:
SUSPICION_MIN_TPR     = 85.0   # TPR individu di bawah ini → flagged
SUSPICION_MAX_MEAN    = 0.72   # Mean cosine di bawah ini → flagged
SUSPICION_MAX_STD     = 0.13  # Std dev di atas ini → embedding tidak stabil → flagged

# ==============================================================================
# PHASE 1: BOOTSTRAPPING ENGINE & WARMUP
# ==============================================================================
print("🚀 [Phase 1/4] Booting Core Engines & Models...")

print(" ├─ 🧠 Inisialisasi Lightweight Face Engine (det+rec only)...")
face_engine = _LightFaceEngine(ctx_id=0, det_size=(320, 320))

print(" ├─ 📦 Loading Person Detector (YOLOv8)...")
yolo_model = YOLO(YOLO_MODEL_PATH)

if not os.path.exists(VECTOR_DATABASE_FILE):
    raise FileNotFoundError(f"❌ Database {VECTOR_DATABASE_FILE} tidak ditemukan!")

with open(VECTOR_DATABASE_FILE, 'rb') as f:
    database_registry = pickle.load(f)

registered_students = sorted(list(database_registry.keys()))
print(f" ├─ 📂 Database: {len(registered_students)} mahasiswa terdaftar.")

print(" ├─ 🧠 Membangun USearch Index...")
usearch_index     = Index(ndim=512, metric="cos")
usearch_id_to_nrp = {}
vector_idx_counter = 0

for student_id, data in database_registry.items():
    for emb in data['embeddings']:
        emb_norm = emb / (np.linalg.norm(emb) + 1e-6)
        usearch_index.add(vector_idx_counter, emb_norm)
        usearch_id_to_nrp[vector_idx_counter] = student_id
        vector_idx_counter += 1

print(f" └─ ✅ USearch Index: {len(usearch_index)} vektor total.")

print("\n🔥 [Phase 2/4] Engine Warmup...")
dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
_ = yolo_model(dummy_img, verbose=False)
_ = face_engine._app.get(dummy_img)
print("✅ Warmup selesai.\n")

# ==============================================================================
# WADAH METRIK — GLOBAL
# ==============================================================================
total_cases       = 0
true_positives    = 0
false_acceptances = 0
false_negatives   = 0

pos_cosine_scores = []
neg_cosine_scores = []
search_latencies  = []
ttfm_frames_records = []

# ==============================================================================
# WADAH METRIK — PER MAHASISWA
# { student_id: { "total": int, "tp": int, "fa": int, "fn": int,
#                 "pos_scores": [], "neg_scores": [], "ttfm_frames": [] } }
# ==============================================================================
per_student_stats = {
    sid: {
        "total": 0, "tp": 0, "fa": 0, "fn": 0,
        "pos_scores": [], "neg_scores": [], "ttfm_frames": []
    }
    for sid in registered_students
}

# ==============================================================================
# PHASE 3: STRESS-TEST LOOP
# ==============================================================================
print(f"🏃 [Phase 3/4] Eksekusi {NUM_ITERATIONS} iterasi randomized simulation...")
start_bench_time = time.time()

for iter_idx in range(1, NUM_ITERATIONS + 1):
    current_batch_order = list(registered_students)
    random.shuffle(current_batch_order)

    for student_id in current_batch_order:
        student_path = os.path.join(BASE_DIR, "Face Database", student_id)
        if not os.path.exists(student_path):
            continue

        all_frames = sorted([
            f for f in os.listdir(student_path)
            if f.startswith("frame_") and f.endswith(".jpg")
        ])
        if not all_frames:
            continue

        random_frame = random.choice(all_frames)
        img = cv2.imread(os.path.join(student_path, random_frame))
        if img is None:
            continue

        yolo_results = yolo_model(img, verbose=False)[0]
        if len(yolo_results.boxes) == 0:
            continue

        faces = face_engine._app.get(img)
        if not faces:
            continue

        face = sorted(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True
        )[0]
        live_embedding = face.embedding
        if live_embedding is None:
            continue

        total_cases += 1
        per_student_stats[student_id]["total"] += 1

        # --- USEARCH QUERY ---
        t_start = time.perf_counter()
        live_emb_norm = live_embedding / (np.linalg.norm(live_embedding) + 1e-6)
        matches = usearch_index.search(live_emb_norm, 1)
        search_latencies.append((time.perf_counter() - t_start) * 1000)

        if len(matches) > 0:
            best_similarity = 1.0 - matches[0].distance
            best_match_id   = usearch_id_to_nrp[matches[0].key]
        else:
            best_match_id   = None
            best_similarity = -1.0

        is_correct_id    = (best_match_id == student_id)
        is_above_threshold = (best_similarity >= SIMILARITY_THRESHOLD)

        # --- KLASIFIKASI ---
        if is_above_threshold:
            if is_correct_id:
                true_positives += 1
                per_student_stats[student_id]["tp"] += 1
                pos_cosine_scores.append(best_similarity)
                per_student_stats[student_id]["pos_scores"].append(best_similarity)

                frame_idx = all_frames.index(random_frame) + 1
                ttfm_frames_records.append(frame_idx)
                per_student_stats[student_id]["ttfm_frames"].append(frame_idx)
            else:
                false_acceptances += 1
                per_student_stats[student_id]["fa"] += 1
                neg_cosine_scores.append(best_similarity)
                per_student_stats[student_id]["neg_scores"].append(best_similarity)
        else:
            false_negatives += 1
            per_student_stats[student_id]["fn"] += 1
            if is_correct_id:
                pos_cosine_scores.append(best_similarity)
                per_student_stats[student_id]["pos_scores"].append(best_similarity)
            else:
                neg_cosine_scores.append(best_similarity)
                per_student_stats[student_id]["neg_scores"].append(best_similarity)

    if iter_idx % 100 == 0:
        print(f" ├─ ⏳ [{iter_idx}/{NUM_ITERATIONS}] iterasi selesai...")

t_total_bench = time.time() - start_bench_time

# ==============================================================================
# PHASE 4: KALKULASI STATISTIK GLOBAL
# ==============================================================================
tpr = (true_positives  / total_cases) * 100 if total_cases > 0 else 0
far = (false_acceptances / total_cases) * 100 if total_cases > 0 else 0
fnr = (false_negatives  / total_cases) * 100 if total_cases > 0 else 0

avg_latency  = np.mean(search_latencies)
max_latency  = np.max(search_latencies)

avg_ttfm_frames = np.mean(ttfm_frames_records) if ttfm_frames_records else 0
avg_ttfm_ms     = (avg_ttfm_frames / 50) * 1000

pos_mean = np.mean(pos_cosine_scores)  if pos_cosine_scores else 0
pos_std  = np.std(pos_cosine_scores)   if pos_cosine_scores else 0
pos_min  = np.min(pos_cosine_scores)   if pos_cosine_scores else 0
pos_max  = np.max(pos_cosine_scores)   if pos_cosine_scores else 0
neg_mean = np.mean(neg_cosine_scores)  if neg_cosine_scores else 0
neg_std  = np.std(neg_cosine_scores)   if neg_cosine_scores else 0

# Separation margin: jarak antara mean pos dan mean neg dibanding threshold
separation_margin = pos_mean - SIMILARITY_THRESHOLD

# ==============================================================================
# PHASE 4A: KALKULASI PER-STUDENT
# ==============================================================================
per_student_report = {}
flagged_students   = []

for sid, s in per_student_stats.items():
    if s["total"] == 0:
        continue

    s_tpr = (s["tp"] / s["total"]) * 100
    s_far = (s["fa"] / s["total"]) * 100
    s_fnr = (s["fn"] / s["total"]) * 100

    s_pos_mean = np.mean(s["pos_scores"]) if s["pos_scores"] else 0.0
    s_pos_std  = np.std(s["pos_scores"])  if s["pos_scores"] else 0.0
    s_pos_min  = np.min(s["pos_scores"])  if s["pos_scores"] else 0.0
    s_pos_max  = np.max(s["pos_scores"])  if s["pos_scores"] else 0.0

    s_ttfm_avg = np.mean(s["ttfm_frames"]) if s["ttfm_frames"] else 0.0

    # Cek jumlah vektor terdaftar (coverage)
    n_registered_vecs = len(database_registry[sid]["embeddings"])

    per_student_report[sid] = {
        "total": s["total"],
        "tpr": s_tpr, "far": s_far, "fnr": s_fnr,
        "pos_mean": s_pos_mean, "pos_std": s_pos_std,
        "pos_min": s_pos_min, "pos_max": s_pos_max,
        "ttfm_avg": s_ttfm_avg,
        "n_vecs": n_registered_vecs,
    }

    # Flagging logic: kumpulkan alasan flag
    reasons = []
    if s_tpr < SUSPICION_MIN_TPR:
        reasons.append(f"TPR {s_tpr:.1f}% < {SUSPICION_MIN_TPR}%")
    if s_pos_mean < SUSPICION_MAX_MEAN and s["pos_scores"]:
        reasons.append(f"Mean sim {s_pos_mean:.4f} < {SUSPICION_MAX_MEAN}")
    if s_pos_std > SUSPICION_MAX_STD and s["pos_scores"]:
        reasons.append(f"Std dev {s_pos_std:.4f} > {SUSPICION_MAX_STD} (embedding tidak stabil)")

    if reasons:
        flagged_students.append((sid, per_student_report[sid], reasons))

# Sort flagged: paling parah (TPR terendah) duluan
flagged_students.sort(key=lambda x: x[1]["tpr"])

# Sort semua mahasiswa by TPR untuk tabel ranking
all_students_sorted = sorted(
    per_student_report.items(),
    key=lambda x: x[1]["tpr"]
)

# ==============================================================================
# OUTPUT LAPORAN
# ==============================================================================
SEP  = "=" * 95
SEP2 = "-" * 95

print("\n" + SEP)
print("📊  LAPORAN VALIDASI ROBUSTNESS VEKTOR — FACESORT PIPELINE")
print(f"    Strategi Seleksi Frame: Anchor (symmetry terbaik) + 24 frame pose diversity tertinggi")
print(SEP)
print(f"🔄  Total Kasus Terproses   : {total_cases:,} iterasi (acak, {NUM_ITERATIONS}x shuffle per mahasiswa)")
print(f"⏱️  Total Waktu Komputasi   : {t_total_bench:.2f} detik")
print(SEP2)

# --- AKURASI GLOBAL ---
print("🎯  METRIK AKURASI GLOBAL")
print(SEP2)
print(f"    True Positive Rate  (TPR) : {tpr:.2f} %   ← akurasi identifikasi benar")
print(f"    False Acceptance Rate (FAR): {far:.2f} %   ← salah identifikasi ke mahasiswa lain")
print(f"    False Negative Rate  (FNR) : {fnr:.2f} %   ← wajah benar tapi ditolak / below threshold")
print(SEP2)

# --- DISTRIBUSI COSINE — GLOBAL ---
print("📐  DISTRIBUSI COSINE SIMILARITY — GLOBAL")
print(SEP2)
print(f"    Valid Match  │  Mean : {pos_mean:.4f}  │  Std Dev : {pos_std:.4f}"
      f"  │  Min : {pos_min:.4f}  │  Max : {pos_max:.4f}")
print(f"    Wrong Match  │  Mean : {neg_mean:.4f}  │  Std Dev : {neg_std:.4f}")
print(f"    Threshold live system        : {SIMILARITY_THRESHOLD:.4f}")
print(f"    Separation margin (mean−thr) : {separation_margin:+.4f}"
      f"  {'✅ AMAN' if separation_margin > 0.04 else '⚠️  TIPIS — pertimbangkan turunkan threshold'}")
print(f"    Titik threshold optimal lab  : {(pos_mean + neg_mean) / 2:.4f}"
      if pos_cosine_scores and neg_cosine_scores else "    Titik threshold optimal lab  : N/A")
print(SEP2)

# --- LATENSI USEARCH ---
print("⚡  LATENSI USEARCH INDEX")
print(SEP2)
print(f"    Rata-rata  : {avg_latency:.4f} ms / query")
print(f"    Peak       : {max_latency:.4f} ms / query")
status_latency = "✅ LULUS (< 20.0 ms)" if avg_latency < 20.0 else "❌ OVER BUDGET"
print(f"    Budget 50 FPS (20ms)  : {status_latency}")
print(SEP2)

# --- TTFM ---
print(f"🏁  TTFM — TIME TO FIRST MATCH")
print(SEP2)
print(f"    Rata-rata frame ke-  : {avg_ttfm_frames:.1f}")
print(f"    Estimasi waktu nyata : {avg_ttfm_ms:.2f} ms  ({avg_ttfm_ms/1000:.2f} detik @ 50 FPS)")
print(SEP2)

# --- TABEL SEMUA MAHASISWA ---
print("\n📋  TABEL METRIK PER MAHASISWA (diurutkan TPR terendah → tertinggi)")
print(SEP2)
header = (f"{'NRP':<20} {'N Tes':>6} {'TPR%':>7} {'FAR%':>6} {'FNR%':>6} "
          f"{'Mean Sim':>9} {'Std Dev':>8} {'Min Sim':>8} {'Max Sim':>8} {'TTFM Fr':>8} {'Vek':>4}")
print(header)
print(SEP2)

for sid, r in all_students_sorted:
    flag_marker = " ⚠️ " if any(sid == f[0] for f in flagged_students) else "    "
    print(
        f"{sid:<20} {r['total']:>6} {r['tpr']:>7.1f} {r['far']:>6.2f} {r['fnr']:>6.2f} "
        f"{r['pos_mean']:>9.4f} {r['pos_std']:>8.4f} {r['pos_min']:>8.4f} "
        f"{r['pos_max']:>8.4f} {r['ttfm_avg']:>8.1f} {r['n_vecs']:>4}{flag_marker}"
    )

print(SEP2)

# --- MAHASISWA FLAGGED ---
print(f"\n🚨  MAHASISWA YANG PERLU PERHATIAN  ({len(flagged_students)} dari {len(per_student_report)} mahasiswa)")
print("    Kriteria flag: TPR < {:.0f}%  ATAU  Mean sim < {:.2f}  ATAU  Std dev > {:.3f}".format(
    SUSPICION_MIN_TPR, SUSPICION_MAX_MEAN, SUSPICION_MAX_STD))
print(SEP2)

if not flagged_students:
    print("    ✅ Tidak ada mahasiswa yang ter-flag. Semua vektor memenuhi standar robustness.")
else:
    for sid, r, reasons in flagged_students:
        print(f"  ⚠️  [{sid}]  TPR={r['tpr']:.1f}%  Mean={r['pos_mean']:.4f}"
              f"  Std={r['pos_std']:.4f}  Min={r['pos_min']:.4f}  Vek={r['n_vecs']}")
        for reason in reasons:
            print(f"       → {reason}")
        # Rekomendasi otomatis
        if r["n_vecs"] < 15:
            print(f"       💡 Rekomendasi: Tambah frame registrasi (saat ini hanya {r['n_vecs']} vektor)")
        elif r["pos_std"] > SUSPICION_MAX_STD:
            print(f"       💡 Rekomendasi: Cek kualitas foto — kemungkinan ada frame blur/occluded")
        elif r["tpr"] < SUSPICION_MIN_TPR:
            print(f"       💡 Rekomendasi: Cek apakah foto registrasi cover variasi pose yang cukup")
        print()

print(SEP)
print("✅  Analisis selesai. Laporan di atas siap dijadikan lampiran evaluasi skripsi.")
print(SEP + "\n")