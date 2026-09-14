from __future__ import annotations

import os
import numpy as np
import librosa
import essentia.standard as es

class MusicVideoAnalyzer:
    def __init__(self, video_path: str, target_sr: int = 22050):
        """
        Loads the audio track from the video file using Essentia's MonoLoader.
        """
        self.video_path = video_path
        self.target_sr = target_sr

        # Load audio using Essentia MonoLoader (runs fast, decodes audio from video directly)
        loader = es.MonoLoader(filename=video_path, sampleRate=target_sr)
        self.audio = loader()
        self.duration = len(self.audio) / target_sr

    def analyze(self) -> dict[str, any]:
        """
        Extracts features and returns global curves downsampled to a standard 10 Hz rate.
        """
        sr = self.target_sr
        y = self.audio

        # Use a hop length of 512. At sr=22050, 512 frames is ~23 ms.
        # Downsample features to a standard 10 Hz (0.1s steps) for alignment and fast scoring.
        hop_length = 512
        frames_per_sec = sr / hop_length  # ~43 Hz
        ds_fps = 10.0  # 10 Hz downsampled rate

        # 1. Core low-level features
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # Mel spectrogram for band-specific energies
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=64)
        mel_freqs = librosa.mel_frequencies(n_mels=64, fmin=0.0, fmax=sr/2.0)

        # Slice frequency bands
        bass_mask = mel_freqs < 150.0
        synth_mask = (mel_freqs >= 500.0) & (mel_freqs <= 3000.0)
        vocal_mask = (mel_freqs >= 200.0) & (mel_freqs <= 4000.0)

        bass_energy = np.mean(melspec[bass_mask, :], axis=0) if np.any(bass_mask) else rms
        synth_energy = np.mean(melspec[synth_mask, :], axis=0) if np.any(synth_mask) else rms
        vocal_energy = np.mean(melspec[vocal_mask, :], axis=0) if np.any(vocal_mask) else rms

        # Chroma for repetition and chorus
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_chroma=12)

        # MFCC for novelty
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

        # 2. Essentia Rhythm tracking (extremely robust beats)
        try:
            rhythm_extractor = es.RhythmExtractor2013()
            bpm, beats, beats_confidence, _, _ = rhythm_extractor(y)
        except Exception:
            # Fallback to librosa beat tracking if Essentia rhythm extractor fails
            bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
            beats_confidence = 1.0

        # 3. Downsampling function to align all curves to 10 Hz
        n_ds_frames = int(np.floor(self.duration * ds_fps))
        time_axis = np.arange(n_ds_frames) / ds_fps

        def resample_to_ds(curve: np.ndarray) -> np.ndarray:
            curve_times = np.arange(len(curve)) / frames_per_sec
            return np.interp(time_axis, curve_times, curve)

        rms_ds = resample_to_ds(rms)
        centroid_ds = resample_to_ds(centroid)
        flatness_ds = resample_to_ds(flatness)
        onset_ds = resample_to_ds(onset_strength)
        bass_ds = resample_to_ds(bass_energy)
        synth_ds = resample_to_ds(synth_energy)
        vocal_ds = resample_to_ds(vocal_energy)

        # Downsample chroma and MFCC
        chroma_ds = np.zeros((12, n_ds_frames))
        for i in range(12):
            chroma_ds[i] = resample_to_ds(chroma[i])

        mfcc_ds = np.zeros((13, n_ds_frames))
        for i in range(13):
            mfcc_ds[i] = resample_to_ds(mfcc[i])

        # Local beat density at 10 Hz
        beat_density_ds = np.zeros(n_ds_frames)
        window_half_width = 3.0  # 6-second window for beat density
        for i, t in enumerate(time_axis):
            local_beats = np.sum((beats >= t - window_half_width) & (beats <= t + window_half_width))
            beat_density_ds[i] = local_beats / (2 * window_half_width)

        # 4. High-level descriptor curves
        # Repeatability / Memorability (using downsampled Chroma SSM)
        repeatability = self._compute_repeatability(chroma_ds, ds_fps)

        # Section Novelty (using MFCC difference)
        novelty = self._compute_novelty(mfcc_ds, ds_fps)

        # Pitch Salience / Melody Salience
        # Estimate: tonal clarity (1.0 - flatness) combined with vocal range energy
        melody_salience = (1.0 - flatness_ds) * vocal_ds

        # Normalize baseline curves to [0, 1]
        def norm(arr: np.ndarray) -> np.ndarray:
            mi, ma = np.min(arr), np.max(arr)
            return (arr - mi) / (ma - mi + 1e-8)

        rms_norm = norm(rms_ds)
        bass_norm = norm(bass_ds)
        onset_norm = norm(onset_ds)
        beat_density_norm = norm(beat_density_ds)
        flux_norm = norm(onset_ds)  # Onset strength is based on spectral difference (flux)
        centroid_norm = norm(centroid_ds)
        flatness_norm = norm(flatness_ds)
        synth_norm = norm(synth_ds)
        vocal_norm = norm(vocal_ds)
        melody_salience_norm = norm(melody_salience)

        return {
            "duration": self.duration,
            "ds_fps": ds_fps,
            "time_axis": time_axis,
            "beats": beats,
            "bpm": bpm,
            "beats_confidence": beats_confidence,
            "rms": rms_norm,
            "bass": bass_norm,
            "onset": onset_norm,
            "beat_density": beat_density_norm,
            "flux": flux_norm,
            "centroid": centroid_norm,
            "flatness": flatness_norm,
            "synth": synth_norm,
            "vocal": vocal_norm,
            "melody_salience": melody_salience_norm,
            "repeatability": repeatability,
            "novelty": novelty,
            "chroma": chroma_ds,
            "mfcc": mfcc_ds,
        }

    def _compute_repeatability(self, chroma: np.ndarray, ds_fps: float, exclude_seconds: float = 15.0) -> np.ndarray:
        n_frames = chroma.shape[1]
        # Normalize chroma features per frame
        chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-6)
        # Cosine self-similarity matrix
        S = np.dot(chroma_norm.T, chroma_norm)

        exclude_frames = int(exclude_seconds * ds_fps)
        repeatability = np.zeros(n_frames)
        for i in range(n_frames):
            left_bound = max(0, i - exclude_frames)
            right_bound = min(n_frames, i + exclude_frames + 1)
            # Sum off-diagonal similarity
            repeatability[i] = np.sum(S[i, :left_bound]) + np.sum(S[i, right_bound:])

        # Normalize
        mi, ma = np.min(repeatability), np.max(repeatability)
        if ma > mi:
            repeatability = (repeatability - mi) / (ma - mi)
        return repeatability

    def _compute_novelty(self, mfcc: np.ndarray, ds_fps: float, window_seconds: float = 5.0) -> np.ndarray:
        n_frames = mfcc.shape[1]
        w_frames = int(window_seconds * ds_fps)
        novelty = np.zeros(n_frames)
        for i in range(w_frames, n_frames - w_frames):
            left_mean = np.mean(mfcc[:, i - w_frames : i], axis=1)
            right_mean = np.mean(mfcc[:, i : i + w_frames], axis=1)
            norm_l = np.linalg.norm(left_mean)
            norm_r = np.linalg.norm(right_mean)
            if norm_l > 0 and norm_r > 0:
                cos_sim = np.dot(left_mean, right_mean) / (norm_l * norm_r)
                novelty[i] = 1.0 - cos_sim
        # Normalize
        mi, ma = np.min(novelty), np.max(novelty)
        if ma > mi:
            novelty = (novelty - mi) / (ma - mi)
        return novelty

    def _find_peaks(self, curve: np.ndarray, threshold: float, min_dist_frames: int) -> list[int]:
        peaks = []
        for i in range(1, len(curve) - 1):
            if curve[i] > curve[i-1] and curve[i] > curve[i+1] and curve[i] > threshold:
                peaks.append(i)
        # Filter peaks to keep highest separated by min_dist_frames
        filtered_peaks = []
        for p in sorted(peaks, key=lambda x: curve[x], reverse=True):
            if all(abs(p - fp) >= min_dist_frames for fp in filtered_peaks):
                filtered_peaks.append(p)
        return sorted(filtered_peaks)

    def detect_anchor_points(self, features: dict[str, any]) -> list[float]:
        """
        Detects musical events/boundaries as anchor points.
        """
        ds_fps = features["ds_fps"]
        time_axis = features["time_axis"]
        rms = features["rms"]
        bass = features["bass"]
        onset = features["onset"]
        beat_density = features["beat_density"]
        novelty = features["novelty"]
        vocal = features["vocal"]

        # First differences
        rms_diff = np.diff(rms, prepend=rms[0])
        bass_diff = np.diff(bass, prepend=bass[0])
        onset_diff = np.diff(onset, prepend=onset[0])
        beat_density_diff = np.diff(beat_density, prepend=beat_density[0])
        vocal_diff = np.diff(vocal, prepend=vocal[0])

        min_dist_frames = int(2.0 * ds_fps)
        anchors = set()

        # Energy peaks
        anchors.update(self._find_peaks(rms, 0.4, min_dist_frames))
        # Energy rises (drops/buildups)
        anchors.update(self._find_peaks(rms_diff, 0.08, min_dist_frames))
        # Bass spikes
        anchors.update(self._find_peaks(bass, 0.4, min_dist_frames))
        anchors.update(self._find_peaks(bass_diff, 0.08, min_dist_frames))
        # Onset peaks (punchy transients)
        anchors.update(self._find_peaks(onset, 0.4, min_dist_frames))
        anchors.update(self._find_peaks(onset_diff, 0.08, min_dist_frames))
        # Novelty peaks (transitions)
        anchors.update(self._find_peaks(novelty, 0.4, min_dist_frames))
        # Vocal entrances
        anchors.update(self._find_peaks(vocal_diff, 0.08, min_dist_frames))

        # Fallback: if very few anchors are found, add grid anchors every 5 seconds
        if len(anchors) < 5:
            grid_times = np.arange(5.0, self.duration - 5.0, 5.0)
            for gt in grid_times:
                anchors.add(int(gt * ds_fps))

        return sorted([time_axis[idx] for idx in anchors])

    def _end_boundary_components(
        self,
        end_idx: int,
        features: dict[str, any],
    ) -> dict[str, float]:
        """
        Scores how suitable a single timeline point is as a clip ending.

        Good endings tend to land near beats, coincide with musical novelty,
        and resolve energy instead of cutting in the middle of a rise.
        """
        ds_fps = features["ds_fps"]
        time_axis = features["time_axis"]
        rms = features["rms"]
        onset = features["onset"]
        novelty = features["novelty"]
        beats = features["beats"]

        safe_idx = max(0, min(end_idx, len(time_axis) - 1))
        end_time = time_axis[safe_idx]

        if len(beats) > 0:
            nearest_beat_distance = min(abs(end_time - beat) for beat in beats)
            beat_score = max(0.0, 1.0 - (nearest_beat_distance / 0.35))
        else:
            beat_score = 0.0

        before_start = max(0, safe_idx - int(2.0 * ds_fps))
        after_end = min(len(rms), safe_idx + int(2.0 * ds_fps))
        before = rms[before_start:safe_idx]
        after = rms[safe_idx:after_end]
        before_energy = float(np.mean(before)) if len(before) > 0 else 0.0
        after_energy = float(np.mean(after)) if len(after) > 0 else before_energy
        release_score = max(0.0, before_energy - after_energy)

        return {
            "beat": float(beat_score),
            "novelty": float(novelty[safe_idx]),
            "release": float(release_score),
            "onset": float(onset[safe_idx]),
        }

    def _end_boundary_score(
        self,
        end_idx: int,
        nominal_end_idx: int,
        features: dict[str, any],
        search_radius_frames: int,
    ) -> float:
        components = self._end_boundary_components(end_idx, features)
        nominal_distance = abs(end_idx - nominal_end_idx)
        target_closeness = max(
            0.0,
            1.0 - (nominal_distance / max(1, search_radius_frames)),
        )
        return (
            0.35 * components["beat"]
            + 0.30 * components["novelty"]
            + 0.20 * components["release"]
            + 0.10 * components["onset"]
            + 0.05 * target_closeness
        )

    def refine_end_index(
        self,
        start_idx: int,
        nominal_end_idx: int,
        features: dict[str, any],
        min_length: float,
        max_length: float,
        search_radius: float = 4.0,
    ) -> int:
        """
        Nudge a candidate end point to a nearby audio boundary.

        The search stays close to the requested duration so this improves
        cut quality without turning the duration model into a free-for-all.
        """
        ds_fps = features["ds_fps"]
        time_axis = features["time_axis"]
        n_frames = len(time_axis)
        radius_frames = max(1, int(round(search_radius * ds_fps)))
        min_end_idx = start_idx + int(round(min_length * ds_fps))
        max_end_idx = start_idx + int(round(max_length * ds_fps))
        search_start = max(min_end_idx, nominal_end_idx - radius_frames)
        search_end = min(n_frames - 1, max_end_idx, nominal_end_idx + radius_frames)

        if search_start > search_end:
            return min(max(nominal_end_idx, 0), n_frames - 1)

        return max(
            range(search_start, search_end + 1),
            key=lambda idx: (
                self._end_boundary_score(idx, nominal_end_idx, features, radius_frames),
                -abs(idx - nominal_end_idx),
            ),
        )

    def score_window(
        self,
        start_idx: int,
        end_idx: int,
        features: dict[str, any],
        min_length: float,
        target_min_length: float,
        target_max_length: float,
        max_length: float
    ) -> dict[str, any]:
        """
        Evaluates a single candidate window, returning a detailed score breakdown.
        """
        ds_fps = features["ds_fps"]
        time_axis = features["time_axis"]
        duration = time_axis[end_idx] - time_axis[start_idx]

        # Extract curves
        rms = features["rms"]
        bass = features["bass"]
        onset = features["onset"]
        beat_density = features["beat_density"]
        flux = features["flux"]
        centroid = features["centroid"]
        flatness = features["flatness"]
        synth = features["synth"]
        vocal = features["vocal"]
        melody_sal = features["melody_salience"]
        repeatability = features["repeatability"]
        novelty = features["novelty"]
        beats = features["beats"]

        # Window slices
        w_rms = rms[start_idx:end_idx]
        w_bass = bass[start_idx:end_idx]
        w_onset = onset[start_idx:end_idx]
        w_beat_density = beat_density[start_idx:end_idx]
        w_flux = flux[start_idx:end_idx]
        w_centroid = centroid[start_idx:end_idx]
        w_flatness = flatness[start_idx:end_idx]
        w_synth = synth[start_idx:end_idx]
        w_vocal = vocal[start_idx:end_idx]
        w_melody_sal = melody_sal[start_idx:end_idx]
        w_repeat = repeatability[start_idx:end_idx]
        w_novelty = novelty[start_idx:end_idx]

        # --- 1. Core Terms ---
        loudness_score = np.mean(w_rms)
        bass_score = np.mean(w_bass)
        punchiness_score = np.mean(w_onset)
        beat_density_score = np.mean(w_beat_density)
        flux_score = np.mean(w_flux)
        dynamic_contrast_score = np.percentile(w_rms, 90) - np.percentile(w_rms, 10)

        # Intro/outro penalty
        intro_outro_penalty = 0.0
        margin = 15.0
        start_time = time_axis[start_idx]
        end_time = time_axis[end_idx]
        if start_time < margin:
            intro_outro_penalty += 0.5 * (1.0 - (start_time / margin))
        if end_time > self.duration - margin:
            intro_outro_penalty += 0.5 * (1.0 - ((self.duration - end_time) / margin))

        # Silence penalty
        silence_penalty = 0.0
        if np.mean(w_rms) < 0.08:
            silence_penalty += 1.0
        half = len(w_rms) // 2
        first_half_rms = np.mean(w_rms[:half])
        second_half_rms = np.mean(w_rms[half:])
        if second_half_rms > 0.1 and first_half_rms / second_half_rms < 0.2:
            silence_penalty += 0.3

        core_score = (
            0.25 * loudness_score +
            0.20 * bass_score +
            0.15 * punchiness_score +
            0.15 * beat_density_score +
            0.15 * flux_score +
            0.10 * dynamic_contrast_score -
            0.40 * silence_penalty -
            0.30 * intro_outro_penalty
        )
        core_score = max(0.0, core_score)

        # --- 2. Advanced Terms ---
        pre_start_idx = max(0, start_idx - int(5.0 * ds_fps))
        if pre_start_idx < start_idx:
            pre_rms = np.mean(rms[pre_start_idx:start_idx])
        else:
            pre_rms = 0.0
        drop_spike = max(0.0, np.max(w_rms[:int(3.0 * ds_fps)]) - pre_rms) if pre_rms > 0 else 0.0
        drop_score = drop_spike * np.max(w_onset)

        chorus_score = np.mean(w_rms) * np.mean(w_beat_density) * np.mean(w_repeat)
        novelty_score = np.max(w_novelty)
        emotional_intensity_score = np.mean(w_rms) * np.mean(w_centroid) * np.mean(w_onset)
        momentum_score = np.mean(w_beat_density) * np.mean(w_onset)
        heaviness_score = np.mean(w_bass) * np.mean(w_rms) * (1.0 - np.mean(w_centroid))
        darkness_score = 1.0 - np.mean(w_centroid)
        aggression_score = np.mean(w_rms) * np.mean(w_flux) * np.mean(w_flatness)
        darkness_aggression_score = darkness_score * aggression_score
        bass_impact_score = np.max(w_onset) * np.mean(w_bass)
        synth_density_score = np.mean(w_synth) * (1.0 - np.mean(w_flatness))
        vocal_presence_score = np.mean(w_vocal) * (1.0 - np.mean(w_flatness)) * np.mean(w_melody_sal)
        melody_salience_score = np.mean(w_melody_sal)
        repeatability_score = np.mean(w_repeat)

        contrast_score = 0.0
        prev_window_start = max(0, start_idx - len(w_rms))
        if prev_window_start < start_idx:
            prev_rms = np.mean(rms[prev_window_start:start_idx])
            prev_centroid = np.mean(centroid[prev_window_start:start_idx])
            contrast_score = 0.5 * abs(np.mean(w_rms) - prev_rms) + 0.5 * abs(np.mean(w_centroid) - prev_centroid)

        advanced_score = (
            0.12 * drop_score +
            0.12 * chorus_score +
            0.08 * novelty_score +
            0.08 * emotional_intensity_score +
            0.08 * momentum_score +
            0.08 * heaviness_score +
            0.06 * darkness_aggression_score +
            0.06 * bass_impact_score +
            0.06 * synth_density_score +
            0.06 * vocal_presence_score +
            0.06 * melody_salience_score +
            0.06 * repeatability_score +
            0.04 * contrast_score
        )
        advanced_score = max(0.0, advanced_score)

        # --- 3. Dynamic Terms ---
        # A. Start Impact
        first_1_5s_frames = max(1, int(1.5 * ds_fps))
        start_impact_score = np.mean(w_onset[:first_1_5s_frames]) * np.mean(w_rms[:first_1_5s_frames])

        # B. Ending Cleanliness
        end_components = self._end_boundary_components(end_idx, features)
        ending_cleanliness_score = (
            0.45 * end_components["beat"]
            + 0.30 * end_components["novelty"]
            + 0.15 * end_components["release"]
            + 0.10 * end_components["onset"]
        )

        # C. Energy Density
        energy_density_score = np.mean(w_rms)

        # D. Arc / Payoff
        peak_idx = np.argmax(w_rms)
        rel_peak_pos = peak_idx / len(w_rms)
        if 0.15 <= rel_peak_pos <= 0.85:
            arc_payoff_score = 1.0
        else:
            arc_payoff_score = max(0.0, 1.0 - min(abs(rel_peak_pos - 0.15), abs(rel_peak_pos - 0.85)) * 4.0)

        # E. Phrase Boundary
        phrase_boundary_score = max(novelty[start_idx], novelty[end_idx])

        # F. Repetition Penalty
        w_chroma = features["chroma"][:, start_idx:end_idx]
        max_repetition_frames = 120
        if w_chroma.shape[1] > max_repetition_frames:
            sample_idx = np.linspace(
                0,
                w_chroma.shape[1] - 1,
                max_repetition_frames,
                dtype=int,
            )
            w_chroma = w_chroma[:, sample_idx]
        if w_chroma.shape[1] > 1:
            w_chroma_norm = w_chroma / (np.linalg.norm(w_chroma, axis=0, keepdims=True) + 1e-6)
            w_S = np.dot(w_chroma_norm.T, w_chroma_norm)
            n_frames = w_S.shape[0]
            rep_sum = np.sum(w_S) - np.trace(w_S)
            repetition_penalty = float(rep_sum / (n_frames * (n_frames - 1)))
        else:
            repetition_penalty = 0.0

        # G. Padding Penalty
        sorted_rms = np.sort(w_rms)
        q25_idx = int(0.25 * len(sorted_rms))
        q25_mean = np.mean(sorted_rms[:q25_idx]) if q25_idx > 0 else sorted_rms[0]
        padding_penalty = max(0.0, 1.0 - (q25_mean / 0.05)) if q25_mean < 0.05 else 0.0

        # Target duration penalties
        if duration < target_min_length:
            padding_penalty += 0.5 * (1.0 - (duration - min_length) / (target_min_length - min_length + 1e-6))
        elif duration > target_max_length:
            padding_penalty += 0.5 * ((duration - target_max_length) / (max_length - target_max_length + 1e-6))

        # H. Short-form Suitability
        short_form_suitability_score = start_impact_score * energy_density_score * arc_payoff_score

        # I. Duration Bonus (weight longer clips slightly higher)
        duration_bonus = 0.10 * ((duration - min_length) / (max_length - min_length + 1e-6))

        # Combine dynamic terms
        dynamic_score = (
            0.15 * start_impact_score +
            0.10 * ending_cleanliness_score +
            0.15 * energy_density_score +
            0.15 * arc_payoff_score +
            0.10 * phrase_boundary_score +
            0.15 * short_form_suitability_score +
            duration_bonus -  # Add positive length bias
            0.15 * repetition_penalty -
            0.15 * padding_penalty
        )
        dynamic_score = max(0.0, dynamic_score)

        # Overall combination
        combined_score = 0.30 * core_score + 0.35 * advanced_score + 0.35 * dynamic_score

        return {
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration": round(duration, 2),
            "combined_score": round(float(combined_score), 4),
            "core_score": round(float(core_score), 4),
            "advanced_score": round(float(advanced_score), 4),
            "breakdown": {
                "loudness": round(float(loudness_score), 3),
                "bass": round(float(bass_score), 3),
                "punchiness": round(float(punchiness_score), 3),
                "beat_density": round(float(beat_density_score), 3),
                "flux": round(float(flux_score), 3),
                "dynamic_contrast": round(float(dynamic_contrast_score), 3),
                "drop_likelihood": round(float(drop_score), 3),
                "chorus_hook": round(float(chorus_score), 3),
                "novelty": round(float(novelty_score), 3),
                "emotional_intensity": round(float(emotional_intensity_score), 3),
                "momentum": round(float(momentum_score), 3),
                "heaviness": round(float(heaviness_score), 3),
                "darkness_aggression": round(float(darkness_aggression_score), 3),
                "bass_impact": round(float(bass_impact_score), 3),
                "synth_density": round(float(synth_density_score), 3),
                "vocal_presence": round(float(vocal_presence_score), 3),
                "melody_salience": round(float(melody_salience_score), 3),
                "repeatability": round(float(repeatability_score), 3),
                "contrast_from_prev": round(float(contrast_score), 3),
                "silence_penalty": round(float(silence_penalty), 3),
                "intro_outro_penalty": round(float(intro_outro_penalty), 3),

                # New terms
                "start_impact": round(float(start_impact_score), 3),
                "ending_cleanliness": round(float(ending_cleanliness_score), 3),
                "end_beat_alignment": round(float(end_components["beat"]), 3),
                "end_novelty": round(float(end_components["novelty"]), 3),
                "end_energy_release": round(float(end_components["release"]), 3),
                "energy_density": round(float(energy_density_score), 3),
                "arc_payoff": round(float(arc_payoff_score), 3),
                "phrase_boundary": round(float(phrase_boundary_score), 3),
                "duration_bonus": round(float(duration_bonus), 3),
                "repetition_penalty": round(float(repetition_penalty), 3),
                "padding_penalty": round(float(padding_penalty), 3),
                "short_form_suitability": round(float(short_form_suitability_score), 3),
            }
        }

    def score_windows(
        self,
        features: dict[str, any],
        dynamic: bool = True,
        min_length: float = 45.0,
        target_min_length: float = 55.0,
        target_max_length: float = 65.0,
        max_length: float = 75.0,
        window_duration: float = 60.0,
        step_duration: float = 1.0,
        refine_endings: bool = True,
        end_search_radius: float = 4.0,
        duration_step: float = 4.0,
        max_dynamic_anchors: int = 80,
    ) -> list[dict[str, any]]:
        """
        Generates and scores candidate windows. Supports dynamic anchor-based windows or fixed-length fallback.
        """
        ds_fps = features["ds_fps"]
        time_axis = features["time_axis"]
        n_frames = len(time_axis)

        candidates = []

        if dynamic:
            # 1. Detect anchor points
            anchors = self.detect_anchor_points(features)
            if max_dynamic_anchors > 0 and len(anchors) > max_dynamic_anchors:
                rms = features["rms"]
                bass = features["bass"]
                onset = features["onset"]
                novelty = features["novelty"]
                vocal = features["vocal"]

                def anchor_score(anchor_time: float) -> float:
                    idx = max(0, min(int(round(anchor_time * ds_fps)), n_frames - 1))
                    return float(
                        0.25 * rms[idx]
                        + 0.20 * bass[idx]
                        + 0.20 * onset[idx]
                        + 0.20 * novelty[idx]
                        + 0.15 * vocal[idx]
                    )

                anchors = sorted(
                    sorted(anchors, key=anchor_score, reverse=True)[:max_dynamic_anchors]
                )

            # 2. Start offsets (how far before anchor we start the window)
            target_length = (target_min_length + target_max_length) / 2.0
            if target_length >= 45.0:
                start_offsets = [0.0, 3.0, 6.0, 10.0]
            else:
                start_offsets = [0.0, 1.0, 2.0, 3.0, 5.0]

            # 3. Durations (coarser by default for long V2 clips)
            durations = np.arange(min_length, max_length + 0.1, duration_step)

            # Keep track of generated intervals to avoid duplicates
            seen_intervals = set()

            for anchor in anchors:
                for offset in start_offsets:
                    start_time = max(0.0, anchor - offset)
                    start_idx = int(round(start_time * ds_fps))

                    for dur in durations:
                        end_idx = start_idx + int(round(dur * ds_fps))
                        if end_idx >= n_frames:
                            continue
                        if refine_endings:
                            end_idx = self.refine_end_index(
                                start_idx,
                                end_idx,
                                features,
                                min_length=min_length,
                                max_length=max_length,
                                search_radius=end_search_radius,
                            )

                        interval = (start_idx, end_idx)
                        if interval in seen_intervals:
                            continue
                        seen_intervals.add(interval)

                        cand = self.score_window(
                            start_idx,
                            end_idx,
                            features,
                            min_length=min_length,
                            target_min_length=target_min_length,
                            target_max_length=target_max_length,
                            max_length=max_length
                        )
                        candidates.append(cand)
        else:
            # Fixed-length fallback (traditional sliding windows)
            win_size = int(window_duration * ds_fps)
            step_size = int(step_duration * ds_fps)
            for start_idx in range(0, n_frames - win_size, step_size):
                end_idx = start_idx + win_size
                cand = self.score_window(
                    start_idx,
                    end_idx,
                    features,
                    min_length=window_duration,
                    target_min_length=window_duration,
                    target_max_length=window_duration,
                    max_length=window_duration
                )
                candidates.append(cand)

        return candidates

    def select_top_clips(
        self,
        candidates: list[dict[str, any]],
        num_clips: int = 5,
        min_gap: float = 15.0,
        overlap_penalty: float = 0.75,
        gap_penalty: float = 0.15,
        duplicate_overlap_threshold: float = 0.92,
    ) -> list[dict[str, any]]:
        """
        Selects top clips using score-ordered soft diversification.

        Overlap is allowed, but each candidate is penalized by how much it
        overlaps already selected clips. Near-identical windows are still
        suppressed so the output does not contain duplicates of the same edit.
        """
        if not candidates or num_clips <= 0:
            return []

        def overlap_ratio(left: dict[str, any], right: dict[str, any]) -> float:
            overlap = max(
                0.0,
                min(left["end_time"], right["end_time"])
                - max(left["start_time"], right["start_time"]),
            )
            if overlap <= 0:
                return 0.0
            shorter_duration = max(1e-6, min(left["duration"], right["duration"]))
            return overlap / shorter_duration

        def gap_closeness(left: dict[str, any], right: dict[str, any]) -> float:
            if min_gap <= 0:
                return 0.0
            if overlap_ratio(left, right) > 0:
                return 0.0
            gap = max(
                0.0,
                max(left["start_time"], right["start_time"])
                - min(left["end_time"], right["end_time"]),
            )
            return max(0.0, (min_gap - gap) / min_gap)

        remaining = sorted(
            candidates,
            key=lambda x: (-x["combined_score"], x["start_time"], x["end_time"]),
        )
        selected = []
        score_scale = max(abs(candidate["combined_score"]) for candidate in remaining)
        score_scale = max(score_scale, 1e-6)

        while remaining and len(selected) < num_clips:
            best_idx = None
            best_key = None
            best_metrics = None

            for idx, candidate in enumerate(remaining):
                max_overlap = max(
                    (overlap_ratio(candidate, clip) for clip in selected),
                    default=0.0,
                )
                if (
                    selected
                    and duplicate_overlap_threshold > 0
                    and max_overlap >= duplicate_overlap_threshold
                ):
                    continue

                max_gap_closeness = max(
                    (gap_closeness(candidate, clip) for clip in selected),
                    default=0.0,
                )
                penalty = score_scale * (
                    overlap_penalty * max_overlap
                    + gap_penalty * max_gap_closeness
                )
                adjusted_score = candidate["combined_score"] - penalty
                key = (
                    adjusted_score,
                    candidate["combined_score"],
                    candidate["duration"],
                    -candidate["start_time"],
                )
                if best_key is None or key > best_key:
                    best_idx = idx
                    best_key = key
                    best_metrics = (adjusted_score, penalty, max_overlap, max_gap_closeness)

            if best_idx is None or best_metrics is None:
                break

            candidate = remaining.pop(best_idx)
            adjusted_score, penalty, max_overlap, max_gap_closeness = best_metrics
            selected_clip = dict(candidate)
            selected_clip["breakdown"] = dict(candidate.get("breakdown", {}))
            selected_clip["selection_score"] = round(float(adjusted_score), 4)
            selected_clip["selection_penalty"] = round(float(penalty), 4)
            selected_clip["selection_overlap_ratio"] = round(float(max_overlap), 3)
            selected_clip["selection_gap_closeness"] = round(float(max_gap_closeness), 3)
            selected.append(selected_clip)

        return sorted(selected, key=lambda x: x["start_time"])
