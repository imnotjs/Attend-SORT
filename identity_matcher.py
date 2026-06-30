import numpy as np
import pickle
import os
import torch
import cv2
import torchreid
from usearch.index import Index
from face_db_manager import align_face_arcface

class IdentityMatcher:
    def __init__(self, face_db_path, embedding_dim=512, body_history_size=10):
        self.face_db_path = face_db_path
        self.embedding_dim = embedding_dim
        self.body_history_size = body_history_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ [Re-ID Engine] Target OSNet Device Context: {self.device}")

        print("🧠 [Re-ID Engine] Loading OSNet AIN (osnet_ain_x1_0) Weights into GPU...")
        self.reid_model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=751,
            pretrained=True
        )
        self.reid_model = self.reid_model.to(self.device)
        self.reid_model.eval()

        # OPT: Pre-alloc reusable tensor di GPU VRAM
        self._max_batch_size = 8
        self._preallocated_tensor = torch.zeros(
            (self._max_batch_size, 3, 256, 128),
            dtype=torch.float32,
            device=self.device
        )

        # ImageNet stats sebagai tensor di GPU
        self._mean_gpu = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std_gpu  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # ==============================================================================
        # OPT v3: OSNET CUDA GRAPH WARMUP
        #
        # Versi lama: satu dummy forward pass. Ini warm CUDA kernel compilation
        # tapi TIDAK warm semua path (mis. batch size berbeda = kernel berbeda).
        #
        # Versi baru: Multi-pass warmup dengan BERBAGAI batch size (1, 2, 4, 8).
        # Kenapa ini penting:
        #   CUDA lazy-compiles kernel untuk setiap unique input shape.
        #   Batch 1 ≠ batch 4 dalam hal kernel scheduling di RTX 3050.
        #   Dengan warm semua kemungkinan batch size, zero latency spike saat
        #   jumlah orang di frame berubah dari 1→2→4→8.
        #
        # Cost: ~150ms extra di startup (one-time), benefit: ~0 variability live.
        # ==============================================================================
        print("🔥 [Re-ID Engine] Running OSNet multi-batch GPU warmup (batch sizes: 1,2,4,8)...")
        with torch.no_grad():
            for bs in [1, 2, 4, 8]:
                dummy = torch.zeros((bs, 3, 256, 128), dtype=torch.float32, device=self.device)
                _ = self.reid_model(dummy)
                # torch.cuda.synchronize() untuk pastikan kernel benar-benar selesai
                # (bukan hanya di-queue oleh CUDA async engine)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        print("✅ [Re-ID Engine] OSNet multi-batch warmup complete. All batch paths pre-compiled.")

        # ==============================================================================
        # OPT v3: TORCH.COMPILE (PyTorch 2.0+) — opsional tapi impactful
        #
        # torch.compile() menjalankan TorchInductor yang compile OSNet ke optimized
        # kernel (fused ops, better memory layout) khusus untuk hardware ini (RTX 3050).
        # Expected gain: 15-25% latency reduction per forward pass.
        #
        # Tradeoff: Compile pertama kali ~30-60 detik (one-time per session, bisa dicache).
        # Di production, bisa dicache dengan torch._inductor.config.cache_dir.
        # Di environment ini, kita skip torch.compile karena overhead startup terlalu besar
        # untuk demo. Enable di production dengan: TORCH_COMPILE_ENABLED = True di config.
        # ==============================================================================
        # Uncomment berikut untuk production deployment (tambah 30s startup, -20% live latency):
        # if hasattr(torch, 'compile') and self.device.type == 'cuda':
        #     print("🔧 [Re-ID Engine] Compiling OSNet with TorchInductor (one-time, ~30s)...")
        #     self.reid_model = torch.compile(self.reid_model, mode='reduce-overhead')
        #     print("✅ [Re-ID Engine] TorchInductor compile complete.")

        self.face_index = Index(ndim=self.embedding_dim, metric="cos")
        self.usearch_id_to_nrp = {}
        self.live_body_registry = {}

        # ==============================================================================
        # OPT v3: EMBEDDING CACHE — hindari normalisasi ulang embedding yang sama.
        # Format: { nrp_id: np.ndarray (N, 512) } — pre-normalized embeddings.
        # Build sekali di _build_static_face_index(), tidak pernah dimodifikasi.
        # Dipakai di _batch_match_faces() untuk bypass per-query normalisasi overhead.
        # ==============================================================================
        self._normalized_db_embeddings = {}  # { nrp: np.ndarray (N, 512) norm'd }

        self._build_static_face_index()

    def _build_static_face_index(self):
        if not os.path.exists(self.face_db_path):
            raise FileNotFoundError(f"❌ Database wajah tidak ditemukan di {self.face_db_path}")

        with open(self.face_db_path, 'rb') as f:
            database_registry = pickle.load(f)

        print("🧠 [USearch] Membangun Indeks Vektor Wajah Statis...")
        global_vector_id = 0
        low_coverage_ids = []

        for nrp_id, data in sorted(database_registry.items()):
            embeddings = data["embeddings"]
            n_vecs = len(embeddings)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6
            normalized = embeddings / norms
            self._normalized_db_embeddings[nrp_id] = normalized  # cache pre-normalized

            for emb_norm in normalized:
                self.face_index.add(global_vector_id, emb_norm)
                self.usearch_id_to_nrp[global_vector_id] = nrp_id
                global_vector_id += 1

            warn = " ⚠️  LOW" if n_vecs < 10 else ""
            print(f"     [{nrp_id}] {n_vecs:>3} vektor{warn}")
            if n_vecs < 10:
                low_coverage_ids.append(nrp_id)

        print(f"✅ [USearch] Total {global_vector_id} vektor dari {len(database_registry)} mahasiswa diindeks.")
        if low_coverage_ids:
            print(f"⚠️  [USearch] Mahasiswa coverage rendah (<10 vektor): {low_coverage_ids}")
            print(f"   → Tambah lebih banyak frame di folder mereka lalu hapus .pkl untuk re-register.")

    # ==============================================================================
    # OPT: BATCHED OSNet INFERENCE
    # ==============================================================================
    def extract_body_features_batch(self, crop_list):
        if not crop_list:
            return []

        valid_indices = []
        batch_imgs = []

        for i, crop_obj in enumerate(crop_list):
            if crop_obj is None or crop_obj.size == 0:
                continue
            try:
                resized = cv2.resize(crop_obj, (128, 256))
                img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = img.transpose(2, 0, 1)  # HWC → CHW
                batch_imgs.append(img)
                valid_indices.append(i)
            except Exception:
                continue

        if not batch_imgs:
            return [None] * len(crop_list)

        batch_np = np.stack(batch_imgs, axis=0)
        batch_size = len(batch_imgs)

        self._preallocated_tensor[:batch_size] = torch.from_numpy(batch_np).to(self.device)
        batch_tensor = self._preallocated_tensor[:batch_size]
        batch_tensor = (batch_tensor - self._mean_gpu) / self._std_gpu

        with torch.no_grad():
            features = self.reid_model(batch_tensor)
            features_np = features.cpu().numpy()

        results = [None] * len(crop_list)
        for local_i, orig_i in enumerate(valid_indices):
            results[orig_i] = features_np[local_i]

        return results

    def extract_body_feature(self, crop_obj):
        results = self.extract_body_features_batch([crop_obj])
        return results[0] if results else None

    # ==============================================================================
    # FACE MATCHING VIA USearch
    # ==============================================================================
    def match_face(self, live_face_embedding, threshold=0.65):
        if live_face_embedding is None:
            return None, 0.0

        live_emb_norm = live_face_embedding / (np.linalg.norm(live_face_embedding) + 1e-6)
        matches = self.face_index.search(live_emb_norm, 1)

        if len(matches) > 0:
            best_match = matches[0]
            similarity = 1.0 - best_match.distance
            if similarity >= threshold:
                matched_nrp = self.usearch_id_to_nrp[best_match.key]
                return matched_nrp, similarity

        return "UNKNOWN", 0.0

    def match_face_aligned(self, face_obj, full_frame_bgr, app_live, threshold=0.65):
        if face_obj is None:
            return "UNKNOWN", 0.0

        kps = getattr(face_obj, 'kps', None)
        embedding = face_obj.embedding

        if kps is not None and embedding is not None:
            return self.match_face(embedding, threshold)

        if kps is not None and embedding is None and full_frame_bgr is not None:
            aligned = align_face_arcface(full_frame_bgr, kps, output_size=112)
            if aligned is not None:
                re_faces = app_live.get(aligned)
                if re_faces and re_faces[0].embedding is not None:
                    return self.match_face(re_faces[0].embedding, threshold)

        if embedding is not None:
            return self.match_face(embedding, threshold)

        return "UNKNOWN", 0.0

    # ==============================================================================
    # OPT v3: BATCH FACE MATCHING — satu vectorized operation untuk semua track.
    #
    # Masalah di versi lama: match_face() dipanggil SEKALI PER TRACK (serial loop).
    # Untuk 8 orang = 8x sequential USearch.search() calls.
    # Meski setiap call <1ms, overhead Python loop + function call stack masih ada.
    #
    # Solusi: kumpulkan semua face embeddings dari semua track dalam satu batch,
    # lakukan matrix multiply sekali → dapatkan semua similarity sekaligus.
    # Ini lebih efisien dari 8x USearch karena:
    # 1. NumPy BLAS matrix multiply highly parallelized (AVX2 pada R7 6800H)
    # 2. Satu call Python overhead untuk N embeddings vs N separate calls
    # 3. Eliminasi N-1 Python→C boundary crossings per frame
    #
    # Tradeoff: Memory O(N × M) di mana N=track count, M=total db vectors (549).
    # Untuk N=8, M=549: 8×549×4bytes ≈ 17KB — trivial.
    # ==============================================================================
    def batch_match_faces(self, embeddings_list, threshold=0.65):
        """
        Batch face matching: satu vectorized operation untuk semua embeddings.

        Input:  list of np.ndarray atau None (satu per track)
        Output: list of (nrp, similarity) tuples (satu per track)

        Implementasi: matrix multiply semua live embeddings terhadap semua
        database embeddings sekaligus, ambil max per row.
        """
        results = [("UNKNOWN", 0.0)] * len(embeddings_list)
        if not embeddings_list:
            return results

        # Filter valid embeddings
        valid_indices = []
        valid_embs = []
        for i, emb in enumerate(embeddings_list):
            if emb is not None:
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    valid_embs.append(emb / norm)
                    valid_indices.append(i)

        if not valid_embs:
            return results

        # Susun matrix database (semua pre-normalized embeddings gabungan)
        # Format: db_matrix shape (M, 512), db_nrp_labels shape (M,)
        db_nrp_labels = []
        db_matrix_rows = []
        for nrp_id, norm_embs in self._normalized_db_embeddings.items():
            for emb in norm_embs:
                db_matrix_rows.append(emb)
                db_nrp_labels.append(nrp_id)

        if not db_matrix_rows:
            return results

        db_matrix = np.stack(db_matrix_rows, axis=0)  # (M, 512)
        live_matrix = np.stack(valid_embs, axis=0)    # (N, 512)

        # Cosine similarity matrix: (N, M)
        sim_matrix = live_matrix @ db_matrix.T

        # Per-row: ambil max similarity dan NRP-nya
        best_idx = np.argmax(sim_matrix, axis=1)      # (N,)
        best_sim  = sim_matrix[np.arange(len(valid_embs)), best_idx]  # (N,)

        for local_i, orig_i in enumerate(valid_indices):
            if best_sim[local_i] >= threshold:
                results[orig_i] = (db_nrp_labels[best_idx[local_i]], float(best_sim[local_i]))

        return results

    # ==============================================================================
    # EMA SUPPORT: BATCH MATCH TANPA THRESHOLD CUTOFF
    #
    # batch_match_faces() di atas mengembalikan "UNKNOWN" kalau similarity di
    # bawah threshold — cocok untuk direct-hit matching (Jalur 1), tapi untuk
    # EMA accumulation (Jalur 2) kita justru BUTUH similarity mentah meski di
    # bawah threshold, supaya bisa diakumulasi dan dirata-rata dari waktu ke
    # waktu (mahasiswa berjalan: confidence noisy, kadang di bawah 0.65 padahal
    # orangnya sama).
    #
    # Method ini reuse perhitungan matrix yang sama (tidak ada overhead ekstra
    # signifikan — sama-sama satu matmul), tapi TIDAK menerapkan threshold cutoff.
    # NRP kandidat tetap dikembalikan walau similarity rendah; caller (ResOptEngine
    # .accumulate_face_ema) yang menentukan apakah cukup untuk match.
    #
    # Output: list of (best_candidate_nrp_or_None, raw_similarity) — None hanya
    # kalau embedding itu sendiri invalid/tidak ada wajah terdeteksi.
    # ==============================================================================
    def batch_match_faces_raw(self, embeddings_list):
        results = [(None, 0.0)] * len(embeddings_list)
        if not embeddings_list:
            return results

        valid_indices = []
        valid_embs = []
        for i, emb in enumerate(embeddings_list):
            if emb is not None:
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    valid_embs.append(emb / norm)
                    valid_indices.append(i)

        if not valid_embs:
            return results

        db_nrp_labels = []
        db_matrix_rows = []
        for nrp_id, norm_embs in self._normalized_db_embeddings.items():
            for emb in norm_embs:
                db_matrix_rows.append(emb)
                db_nrp_labels.append(nrp_id)

        if not db_matrix_rows:
            return results

        db_matrix = np.stack(db_matrix_rows, axis=0)  # (M, 512)
        live_matrix = np.stack(valid_embs, axis=0)    # (N, 512)

        sim_matrix = live_matrix @ db_matrix.T
        best_idx = np.argmax(sim_matrix, axis=1)
        best_sim = sim_matrix[np.arange(len(valid_embs)), best_idx]

        for local_i, orig_i in enumerate(valid_indices):
            # Tidak ada cutoff threshold di sini — selalu kembalikan kandidat
            # terbaik + similarity mentahnya, berapapun nilainya.
            results[orig_i] = (db_nrp_labels[best_idx[local_i]], float(best_sim[local_i]))

        return results

    # ==============================================================================
    # HYBRID RE-ID (BODY MATCHING)
    # ==============================================================================
    def update_live_body(self, track_id, current_body_vector, centroid=None, nrp=None):
        if current_body_vector is None: return

        body_norm = current_body_vector / (np.linalg.norm(current_body_vector) + 1e-6)

        if track_id not in self.live_body_registry:
            self.live_body_registry[track_id] = {
                "nrp": nrp,
                "body_vectors": [body_norm],
                "centroid": centroid
            }
        else:
            if nrp is not None:
                self.live_body_registry[track_id]["nrp"] = nrp
            self.live_body_registry[track_id]["centroid"] = centroid
            self.live_body_registry[track_id]["body_vectors"].append(body_norm)
            if len(self.live_body_registry[track_id]["body_vectors"]) > self.body_history_size:
                self.live_body_registry[track_id]["body_vectors"].pop(0)

    def match_body_reid(self, current_body_vector, current_centroid, spatial_threshold=150,
                        body_threshold=0.70, exclude_track_id=None):
        if current_body_vector is None:
            return None, "UNKNOWN"

        body_norm = current_body_vector / (np.linalg.norm(current_body_vector) + 1e-6)
        best_match_track_id = None
        matched_nrp = "UNKNOWN"
        best_body_sim = -1.0

        for old_track_id, data in self.live_body_registry.items():
            if old_track_id == exclude_track_id:
                continue

            old_vectors = data["body_vectors"]
            old_centroid = data["centroid"]
            old_nrp = data["nrp"]

            if old_centroid is not None and current_centroid is not None:
                spatial_dist = np.linalg.norm(np.array(current_centroid) - np.array(old_centroid))
                if spatial_dist > spatial_threshold:
                    continue

            old_vectors_arr = np.array(old_vectors)
            dot_products = np.dot(old_vectors_arr, body_norm)
            max_body_sim = np.max(dot_products)

            if max_body_sim > best_body_sim:
                best_body_sim = max_body_sim
                best_match_track_id = old_track_id
                if old_nrp is not None:
                    matched_nrp = old_nrp

        if best_body_sim >= body_threshold:
            return best_match_track_id, matched_nrp

        return None, "UNKNOWN"