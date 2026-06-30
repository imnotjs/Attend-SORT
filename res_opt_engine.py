import time

class ResOptEngine:
    def __init__(self, fps_target=50):
        self.fps_target = fps_target
        self.frame_interval = 1.0 / fps_target

        # ==============================================================================
        # INFERENCE THROTTLE CONSTANTS
        #
        # Dua mode throttle untuk AI thread:
        #
        # IDLE_FPS (25fps → 40ms/cycle):
        #   Aktif saat SEMUA track LOW priority (semua sudah absen) atau tidak ada
        #   track sama sekali. GPU punya ~75% idle time per cycle → clock maintained
        #   di mid-sustained level, tidak throttle, tidak gosong.
        #   Kenapa 25 bukan 45?
        #   - 45fps idle → GPU masih ~55% idle. GPU modern (khususnya laptop) bisa
        #     mulai throttle saat duty cycle >50% sustained → clock drops 10-20%.
        #   - 25fps idle → GPU ~75% idle. Clock jauh lebih stabil di base sustained
        #     frequency. Saat HIGH priority burst masuk, clock naik dari sustained
        #     bukan dari throttled → TTFM lebih cepat 15-30ms pada burst pertama.
        #   - Dari perspektif latency pengguna: idle berarti semua orang sudah
        #     dikenali, tidak ada penalti TTFM untuk mode ini.
        #
        # HIGH PRIORITY = NO THROTTLE (0ms sleep):
        #   Saat ada track yang belum dikenali, AI thread full blast.
        #   TTFM race: tiap ms yang dibuang untuk sleep = TTFM naik.
        #   Window HIGH priority biasanya singkat (<3 detik) → thermal spike
        #   acceptable dan tidak sempat menyebabkan sustained throttle.
        #
        # CAMERA_FPS adalah target kamera dan recording — BUKAN target AI inference.
        # GUI render mengikuti CAMERA_FPS (lewat CameraThread), bukan AI rate.
        # ==============================================================================
        self.CAMERA_FPS              = fps_target          # 50fps — kamera & recording
        self.THROTTLE_IDLE_FPS       = 25                  # AI cycle saat semua LOW priority
        self.THROTTLE_IDLE_INTERVAL  = 1.0 / self.THROTTLE_IDLE_FPS   # 40ms
        # HIGH priority → tidak ada throttle (interval = 0), jadi tidak ada konstanta

        self.priority_registry = {}

        self.pipeline_benchmarks = {
            "yolo_time": [],
            "botsort_time": [],
            "face_det_time": [],
            "face_rec_time": [],
            "osnet_time": []
        }

        # ==============================================================================
        # OPT v3: ADAPTIVE TTFM TRACKER
        #
        # Sebelumnya hanya track stranger_frame_count (berapa kali face det jalan tapi
        # tidak match). Versi baru juga track confidence_history per track untuk
        # DYNAMIC THRESHOLD ADAPTATION.
        #
        # Insight: InsightFace confidence varies dengan kualitas wajah (jarak, pose,
        # blur). Menggunakan fixed threshold 0.65 untuk semua kondisi suboptimal:
        # - Orang dekat, frontal: confidence biasanya 0.72-0.85 → threshold terlalu
        #   konservatif, match harusnya terjadi lebih cepat.
        # - Orang jauh, agak miring: confidence 0.60-0.67 → threshold terlalu agresif,
        #   bisa false match.
        #
        # Solusi: track rolling max confidence. Jika dalam 5 frame terakhir ada ≥3
        # yang ≥0.60 (hampir match), turunkan threshold ke 0.60 sementara untuk
        # window berikutnya. Ini "confidence window voting" — butuh konsistensi
        # bukan hanya satu spike, tapi tidak harus 100% confident.
        # Threshold kembali ke 0.65 setelah match berhasil atau setelah 30 frame.
        # ==============================================================================
        # Format: { track_id: deque of recent confidence scores }
        self._confidence_history = {}
        self._CONFIDENCE_HISTORY_LEN = 6
        self._CONFIDENCE_SOFTEN_THRESHOLD = 0.60
        self._CONFIDENCE_SOFTEN_MIN_HITS = 3
        self._CONFIDENCE_SOFTEN_WINDOW = 30  # reset setelah 30 frame tanpa match

        # ==============================================================================
        # EMA FACE RECOGNITION ACCUMULATOR (khusus mahasiswa BERJALAN)
        #
        # Masalah: mahasiswa yang berjalan punya confidence per-frame yang noisy
        # akibat motion blur + pose berubah cepat. Satu frame bisa dapat angle
        # buruk (confidence jatuh ke 0.55) padahal frame sebelum/sesudahnya bagus
        # (0.68, 0.70). Kalau hanya andalkan "tembak langsung" per frame, orang
        # ini bisa gagal match terus padahal identitasnya konsisten.
        #
        # Solusi: DUA JALUR PARALEL, bukan ganti direct match dengan EMA.
        #   Jalur 1 — DIRECT HIT: raw_similarity >= 0.65 di SATU frame → match
        #             langsung, tidak ada delay. Ini untuk kasus mudah (diam,
        #             frontal, dekat kamera) — TTFM tidak boleh dikorbankan.
        #   Jalur 2 — EMA ACCUMULATION: kalau direct hit gagal, similarity
        #             (walau di bawah 0.65) diakumulasi ke EMA per kandidat NRP.
        #             Setelah cukup sample (4-5 akumulasi), kalau EMA >= 0.62,
        #             declare match. Threshold EMA lebih rendah dari direct
        #             (0.62 vs 0.65) karena EMA sudah meredam noise — angka yang
        #             stabil secara matematis lebih bisa dipercaya dibanding satu
        #             spike sesaat di angka yang sama.
        #
        # EMA per-NRP, BUKAN per-similarity-score generik:
        #   Kalau cuma EMA-kan "skor tertinggi" tanpa peduli siapa NRP-nya, bisa
        #   salah gabung dua orang berbeda yang sekilas mirip di frame berbeda.
        #   Jadi kita track candidate_nrp (NRP dengan similarity tertinggi di
        #   frame ini) dan EMA hanya dihitung selama candidate_nrp KONSISTEN.
        #
        # Switch candidate (kandidat NRP berubah dari sebelumnya):
        #   Reset akumulasi, mulai hitung ulang dari kandidat baru. Ini paling
        #   aman — kandidat yang goyang-goyang antar NRP berarti sinyal belum
        #   solid, lebih baik restart daripada averaging dua identitas berbeda.
        #
        # EMA formula: ema_new = alpha * sample + (1 - alpha) * ema_old
        #   alpha = 0.4 dipilih untuk window efektif ~4-5 sample (2/(N+1) dengan
        #   N=4 → 0.4). Alpha lebih tinggi dari smoothing standar karena pose
        #   orang berjalan berubah cepat — histori >5 frame lalu (100ms+ pada
        #   25fps idle) kurang relevan, EMA harus responsif ke histori terbaru.
        # ==============================================================================
        self._ema_face_state = {}  # { track_id: {candidate_nrp, ema_score, sample_count} }
        self._EMA_ALPHA              = 0.4
        self._EMA_MIN_SAMPLES         = 4     # minimum akumulasi sebelum EMA boleh declare match
        self._EMA_MATCH_THRESHOLD     = 0.62  # threshold setelah EMA matang (vs 0.65 direct)
        self._EMA_DIRECT_THRESHOLD    = 0.65  # threshold jalur 1 (tembak langsung, satu frame)

        # ==============================================================================
        # POSE-WEIGHTED EMA (kasus mahasiswa berjalan MENYAMPING, bukan mendekat)
        #
        # Masalah: mahasiswa yang lewat dari kanan↔kiri hanya punya jendela sempit
        # di tengah frame dengan wajah frontal. Sisanya pose 3/4 atau profil ringan,
        # yang sebenarnya MASIH bisa dideteksi InsightFace, tapi confidence-nya
        # secara sistematis lebih rendah (bukan noise acak, tapi bias dari pose).
        #
        # Solusi: bobot kontribusi sample ke EMA berdasarkan seberapa frontal wajah
        # tersebut, BUKAN buang sample non-frontal sama sekali. Sample profil berat
        # tetap masuk (mempertahankan sample_count, mencegah reset palsu), tapi
        # pengaruhnya ke ema_score dikecilkan lewat alpha efektif yang diskalakan:
        #
        #   pose_weight        = seberapa frontal wajah (1.0 = frontal, →0.0 = profil)
        #   effective_alpha    = _EMA_ALPHA * pose_weight
        #   ema_new            = effective_alpha * sample + (1-effective_alpha) * ema_old
        #
        # Efek: sample frontal mendominasi EMA (alpha mendekati 0.4, responsif).
        # Sample profil berat hampir tidak mengubah EMA (alpha mendekati 0.4*0.3=0.12),
        # tapi tetap menambah sample_count — jadi orang yang sebagian besar framenya
        # miring tapi punya beberapa frame frontal yang solid masih bisa matang.
        #
        # pose_weight dihitung dari face keypoints (kps): 5 titik landmark
        # (mata kiri, mata kanan, hidung, mulut kiri, mulut kanan) yang InsightFace
        # selalu kembalikan bersama embedding. Lihat estimate_pose_weight().
        # ==============================================================================
        self._POSE_WEIGHT_FLOOR = 0.25  # minimum weight untuk profil terberat (jangan 0 — tetap kontribusi sedikit)

        # ==============================================================================
        # VELOCITY-AWARE CROP BOOST (kasus mahasiswa BARU masuk frame dari samping,
        # masih jauh dari kamera secara lateral)
        #
        # Masalah: crop wajah < 40px tinggi → face det di-skip total (run_face=False).
        # Untuk mahasiswa berjalan menyamping yang baru masuk dari tepi frame, ini
        # membuang waktu akumulasi EMA yang seharusnya bisa dimulai lebih awal.
        #
        # Solusi: kalau BoTSort velocity (vx) menunjukkan track BERGERAK MENUJU
        # TENGAH frame (bukan menjauh ke tepi), beri exception — tetap coba face det
        # dengan upscale lebih agresif (4x, dari biasanya 3x untuk zona ini) alih-alih
        # full skip. Track yang menjauh ke tepi TIDAK diberi exception ini, karena
        # akan keluar frame sebentar lagi — usaha ekstra tidak worth it.
        #
        # Threshold velocity dalam px/frame-AI di centroid x-axis. Pada resolusi
        # 1280px, orang berjalan normal melintasi frame dalam ~2-3 detik → di 25fps
        # idle, itu ~12-19px/frame; di HIGH priority full blast (>50fps), bisa lebih
        # kecil per frame karena sampling lebih rapat. Threshold 3.0 px/frame cukup
        # rendah untuk menangkap gerakan lateral nyata tanpa false-positive dari
        # jitter bbox kecil saat orang relatif diam.
        # ==============================================================================
        self._VELOCITY_TOWARD_CENTER_PX = 3.0
        self._BOOST_UPSCALE_FACTOR      = 4   # vs 3x normal untuk zona face_crop_h<60
        self._BOOST_MIN_CROP_H          = 18  # di bawah ini, upscale 4x pun tidak akan reliable

    def register_new_track(self, track_id, current_frame_idx=0):
        if track_id not in self.priority_registry:
            self.priority_registry[track_id] = {
                "priority": "HIGH",
                "status_absen": False,
                "last_body_extract_frame_idx": current_frame_idx,
                "face_skip_counter": 0,
                "nrp_name": None,
                "stranger_frame_count": 0,
                "first_seen_frame_idx": current_frame_idx,
                # OPT v3: tambahan fields untuk adaptive threshold
                "_softened_threshold_since": None,  # frame_idx kapan threshold dilunakkan
            }
        if track_id not in self._confidence_history:
            self._confidence_history[track_id] = []
        if track_id not in self._ema_face_state:
            self._ema_face_state[track_id] = {
                "candidate_nrp": None,
                "ema_score": 0.0,
                "sample_count": 0
            }

    def set_track_as_verified(self, track_id):
        if track_id in self.priority_registry:
            self.priority_registry[track_id]["status_absen"] = True
            self.priority_registry[track_id]["priority"] = "LOW"
        # Reset confidence history saat verified
        if track_id in self._confidence_history:
            self._confidence_history[track_id] = []
        # Reset EMA state saat verified — tidak perlu lanjut akumulasi
        if track_id in self._ema_face_state:
            self._ema_face_state[track_id] = {
                "candidate_nrp": None,
                "ema_score": 0.0,
                "sample_count": 0
            }

    def increment_stranger_counter(self, track_id, weight: float = 1.0):
        if track_id in self.priority_registry:
            self.priority_registry[track_id]["stranger_frame_count"] += weight
            return self.priority_registry[track_id]["stranger_frame_count"]
        return 0.0

    # ==============================================================================
    # OPT v3: ADAPTIVE THRESHOLD LOGIC
    #
    # Dipanggil setelah setiap face detection attempt (termasuk yang gagal).
    # Return threshold yang harus dipakai untuk face matching di frame ini.
    # ==============================================================================
    def get_adaptive_threshold(self, track_id, raw_similarity, current_frame_idx,
                                base_threshold=0.65):
        """
        Adaptive matching threshold berdasarkan riwayat confidence track ini.

        Logic:
        1. Catat raw_similarity ke history (bahkan yang di bawah threshold).
        2. Hitung berapa banyak recent similarities yang ≥ soften_threshold (0.60).
        3. Jika ≥ 3 dari 6 terakhir, anggap orang ini "hampir match" tapi lighting/
           pose tidak ideal → turunkan threshold ke 0.60 untuk window berikutnya.
        4. Auto-reset setelah 30 frame tanpa match (orang mungkin berganti posisi).

        Return: float threshold yang harus dipakai caller untuk match_face()
        """
        if track_id not in self._confidence_history:
            self._confidence_history[track_id] = []

        # Append confidence (0.0 jika face tidak terdeteksi)
        history = self._confidence_history[track_id]
        history.append(raw_similarity)
        if len(history) > self._CONFIDENCE_HISTORY_LEN:
            history.pop(0)

        # Check softening condition
        near_hits = sum(1 for s in history if s >= self._CONFIDENCE_SOFTEN_THRESHOLD)
        track_state = self.priority_registry.get(track_id, {})

        if near_hits >= self._CONFIDENCE_SOFTEN_MIN_HITS:
            track_state["_softened_threshold_since"] = current_frame_idx
            return self._CONFIDENCE_SOFTEN_THRESHOLD

        # Auto-reset softened threshold setelah _CONFIDENCE_SOFTEN_WINDOW frames
        softened_since = track_state.get("_softened_threshold_since")
        if softened_since is not None:
            if (current_frame_idx - softened_since) > self._CONFIDENCE_SOFTEN_WINDOW:
                track_state["_softened_threshold_since"] = None
            else:
                # Masih dalam softened window
                return self._CONFIDENCE_SOFTEN_THRESHOLD

        return base_threshold

    # ==============================================================================
    # EMA FACE RECOGNITION — JALUR 2 (akumulasi untuk mahasiswa berjalan)
    #
    # Dipanggil SETELAH direct match (batch_match_faces) gagal di frame ini,
    # dengan raw similarity terhadap kandidat NRP terbaik (meski di bawah 0.65).
    #
    # Caller flow yang diharapkan:
    #   1. matched_nrp, sim = batch_match_faces(...)   ← Jalur 1: direct hit
    #   2. if matched_nrp != "UNKNOWN": DECLARE MATCH LANGSUNG (tidak panggil ini)
    #   3. else: panggil accumulate_face_ema(track_id, best_candidate_nrp, best_sim,
    #            pose_weight) — pose_weight dari estimate_pose_weight(kps)
    #
    # POSE WEIGHTING (untuk kasus berjalan menyamping):
    #   pose_weight menyesuaikan seberapa besar sample ini boleh mengubah EMA.
    #   Sample frontal (weight≈1.0) → alpha efektif mendekati _EMA_ALPHA (0.4),
    #   responsif penuh. Sample profil berat (weight≈_POSE_WEIGHT_FLOOR=0.25)
    #   → alpha efektif kecil (≈0.1), kontribusi minimal tapi TETAP menambah
    #   sample_count — supaya orang yang sebagian besar frame-nya miring tapi
    #   ada beberapa frame frontal solid masih bisa matang tanpa reset palsu.
    #
    # Return: (matched_nrp_or_None, ema_score)
    #   matched_nrp_or_None bukan None HANYA kalau EMA sudah matang (sample_count
    #   >= _EMA_MIN_SAMPLES) DAN ema_score >= _EMA_MATCH_THRESHOLD (0.62).
    # ==============================================================================
    def accumulate_face_ema(self, track_id, candidate_nrp, raw_similarity, pose_weight=1.0):
        if track_id not in self._ema_face_state:
            self._ema_face_state[track_id] = {
                "candidate_nrp": None,
                "ema_score": 0.0,
                "sample_count": 0
            }

        # Clamp pose_weight ke range aman — jangan biarkan caller kirim nilai
        # di luar [floor, 1.0] yang bisa membuat alpha efektif negatif/>1.
        pose_weight = max(self._POSE_WEIGHT_FLOOR, min(1.0, pose_weight))

        state = self._ema_face_state[track_id]

        # candidate_nrp None artinya tidak ada wajah terdeteksi sama sekali di
        # frame ini (face det gagal) — jangan reset, jangan akumulasi, biarkan
        # state tetap supaya gap satu-dua frame (occlusion sesaat) tidak
        # menghanguskan akumulasi yang sudah terbentuk.
        if candidate_nrp is None:
            return None, state["ema_score"]

        # Switch candidate: NRP kandidat berubah dari sebelumnya → reset akumulasi.
        # Paling aman — kandidat yang goyang antar identitas berarti sinyal belum
        # solid, mending mulai ulang daripada averaging dua orang berbeda.
        if state["candidate_nrp"] is not None and state["candidate_nrp"] != candidate_nrp:
            state["candidate_nrp"]  = candidate_nrp
            state["ema_score"]      = raw_similarity
            state["sample_count"]   = 1
            return None, state["ema_score"]

        # Sample pertama untuk kandidat ini — pose_weight tidak relevan di sini
        # karena belum ada ema_old untuk di-blend, nilai awal = sample mentah.
        if state["candidate_nrp"] is None:
            state["candidate_nrp"] = candidate_nrp
            state["ema_score"]     = raw_similarity
            state["sample_count"]  = 1
        else:
            # Update EMA dengan alpha efektif yang diskalakan pose_weight:
            # ema_new = (base_alpha * pose_weight) * sample + (1 - itu) * ema_old
            effective_alpha = self._EMA_ALPHA * pose_weight
            state["ema_score"] = (
                effective_alpha * raw_similarity +
                (1 - effective_alpha) * state["ema_score"]
            )
            state["sample_count"] += 1

        if (state["sample_count"] >= self._EMA_MIN_SAMPLES and
                state["ema_score"] >= self._EMA_MATCH_THRESHOLD):
            return candidate_nrp, state["ema_score"]

        return None, state["ema_score"]

    # ==============================================================================
    # POSE WEIGHT ESTIMATION dari InsightFace keypoints (kps)
    #
    # InsightFace selalu kembalikan 5 keypoints bersama embedding (urutan standar):
    #   kps[0] = mata kiri, kps[1] = mata kanan, kps[2] = hidung,
    #   kps[3] = mulut kiri, kps[4] = mulut kanan
    # (kiri/kanan dari sudut pandang subjek, bukan kamera — tapi ini tidak penting
    # untuk perhitungan simetri, hanya butuh label konsisten kiri vs kanan)
    #
    # Insight geometris: di wajah frontal, jarak horizontal mata-kiri→hidung dan
    # hidung→mata-kanan HAMPIR SAMA (wajah simetris terhadap garis vertikal hidung).
    # Saat kepala menoleh (yaw), satu sisi "mengecil" akibat perspektif foreshortening
    # — sisi yang menjauh dari kamera tampak lebih dekat ke hidung dalam projeksi 2D.
    #
    #   symmetry_ratio = min(d_left, d_right) / max(d_left, d_right)
    #   1.0  → simetris sempurna → frontal
    #   →0.0 → satu sisi nyaris hilang → profil penuh
    #
    # pose_weight = symmetry_ratio, di-clamp ke _POSE_WEIGHT_FLOOR sebagai lantai.
    #
    # Ini BUKAN pose estimation akurat (tidak ada head pose 3D), tapi proxy murah
    # yang tidak butuh model tambahan — cukup akurat untuk tujuan WEIGHTING
    # (bukan filtering keras), karena kesalahan kecil di estimasi hanya mengubah
    # bobot kontribusi EMA, tidak menentukan match/tidak-match secara langsung.
    # ==============================================================================
    @staticmethod
    def estimate_pose_weight(kps, floor=0.25):
        if kps is None or len(kps) < 5:
            return 1.0  # tidak ada keypoints → asumsikan frontal (tidak menghukum)

        try:
            eye_l  = kps[0]
            eye_r  = kps[1]
            nose   = kps[2]

            d_left  = abs(float(nose[0]) - float(eye_l[0]))
            d_right = abs(float(eye_r[0]) - float(nose[0]))

            if d_left < 1e-3 and d_right < 1e-3:
                return 1.0  # degenerate, jangan divide-by-zero

            symmetry_ratio = min(d_left, d_right) / (max(d_left, d_right) + 1e-6)
            return max(floor, min(1.0, symmetry_ratio))
        except (TypeError, IndexError, ValueError):
            return 1.0  # gagal parse kps → fallback aman, jangan crash pipeline

    # ==============================================================================
    # VELOCITY-AWARE CROP BOOST — cek apakah track bergerak MENUJU tengah frame
    #
    # Dipanggil di Phase A (sebelum decide skip/upscale crop) untuk track dengan
    # crop_h < 40px yang NORMALNYA di-skip total. Kalau track ini terdeteksi
    # bergerak menuju tengah (akan makin frontal terhadap kamera sebentar lagi),
    # beri exception: tetap coba face det dengan upscale lebih agresif.
    #
    # vx diambil dari self._last_known_boxes (dikelola oleh caller di main_script,
    # ResOptEngine tidak menyimpan velocity sendiri — hanya konsumsi & evaluasi).
    #
    # Return: bool — True kalau exception boost harus diberikan.
    # ==============================================================================
    def is_moving_toward_center(self, centroid_x, vx, frame_width):
        frame_center_x = frame_width / 2.0
        is_left_of_center = centroid_x < frame_center_x

        if is_left_of_center:
            return vx > self._VELOCITY_TOWARD_CENTER_PX
        else:
            return vx < -self._VELOCITY_TOWARD_CENTER_PX

    def remove_track(self, track_id):
        if track_id in self.priority_registry:
            del self.priority_registry[track_id]
        if track_id in self._confidence_history:
            del self._confidence_history[track_id]
        if track_id in self._ema_face_state:
            del self._ema_face_state[track_id]

    def evaluate_gatekeepers(self, track_id, current_frame_idx):
        """
        Menentukan apakah track_id ini boleh run face det & body extract.
        Returns: (bool_run_face, bool_run_body)

        OPT v3 changes:
        - HIGH priority face: tetap setiap frame (tidak berubah — sudah optimal)
        - HIGH priority body: dynamic interval berdasarkan frames_alive
          * Fresh track (<40 frame): interval 8 (prioritas ke face)
          * Normal track: interval 4
          * Near-stranger track (>20 stranger frames): interval 6
            → Jika orang hampir di-label UNKNOWN STRANGER, kita reduce body
            extract untuk kasih lebih banyak GPU budget ke face det yang mungkin
            masih bisa match di frame berikutnya.
        - LOW priority body: interval 60 (dari 45)
          → Orang sudah diabsen, body update sangat jarang, GPU budget lebih banyak
          untuk HIGH priority tracks (orang baru yang belum teridentifikasi).
        """
        if track_id not in self.priority_registry:
            self.register_new_track(track_id)

        track_state = self.priority_registry[track_id]

        # --- LOW PRIORITY (SUDAH DIABSEN) ---
        if track_state["priority"] == "LOW":
            run_face = False
            # OPT v3: Naikkan dari 45 ke 60 untuk lebih banyak budget ke HIGH tracks
            run_body = (current_frame_idx % 60 == 0)
            return run_face, run_body

        # --- HIGH PRIORITY (BELUM DIABSEN) ---
        elif track_state["priority"] == "HIGH":
            # Face det setiap frame — sudah optimal dengan LightFaceEngine
            run_face = True

            first_seen = track_state.get("first_seen_frame_idx", current_frame_idx)
            frames_alive = current_frame_idx - first_seen
            stranger_count = track_state.get("stranger_frame_count", 0)

            # Dynamic body interval
            if frames_alive < 40:
                body_interval = 8   # Fresh track: prioritas ke face
            elif stranger_count > 20:
                body_interval = 6   # Near-stranger: kurangi body, beri budget ke face
            else:
                body_interval = 4   # Normal: balance face + body

            last_extract = track_state["last_body_extract_frame_idx"]
            if (current_frame_idx - last_extract) >= body_interval:
                run_body = True
                self.priority_registry[track_id]["last_body_extract_frame_idx"] = current_frame_idx
            else:
                run_body = False

            return run_face, run_body

        return False, False

    # ==============================================================================
    # PIPELINE LATENCY MONITORING
    # ==============================================================================
    def log_pipeline_time(self, component_name, delta_time_ms):
        if component_name in self.pipeline_benchmarks:
            self.pipeline_benchmarks[component_name].append(delta_time_ms)

    def get_average_pipeline_latencies(self):
        avg_latencies = {}
        for component, records in self.pipeline_benchmarks.items():
            avg_latencies[component] = sum(records) / len(records) if records else 0.0
        return avg_latencies