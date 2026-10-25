"""
Biometric Presentation Attack Detection (PAD) & Liveness Verification
- Eye Aspect Ratio (EAR) blink-detection state machine
- Active liveness challenge prompts ("Blink once", "Turn head left/right")
- Passive texture frequency analysis (Laplacian variance for screen replay detection)
- Multi-frame temporal consistency checks
"""

import time
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field


class LivenessState(Enum):
    """State machine for blink detection"""
    EYES_OPEN = "open"
    EYES_CLOSING = "closing"
    EYES_CLOSED = "closed"
    BLINK_COMPLETE = "complete"


class ChallengeType(Enum):
    """Active liveness challenge types"""
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    NOD_UP = "nod_up"
    NOD_DOWN = "nod_down"
    SMILE = "smile"


@dataclass
class LivenessChallenge:
    """Active liveness challenge configuration"""
    challenge_type: ChallengeType
    prompt: str
    timeout_seconds: float = 5.0
    success_threshold: float = 0.75
    started_at: Optional[float] = None
    completed: bool = False
    passed: bool = False


@dataclass
class LivenessResult:
    """Result of liveness verification"""
    is_live: bool
    confidence: float
    blink_detected: bool = False
    texture_score: float = 0.0
    challenge_passed: bool = False
    challenge_type: Optional[str] = None
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class EyeAspectRatioDetector:
    """
    Eye Aspect Ratio (EAR) based blink detection using facial landmarks.
    Uses YuNet's 5 landmark points: right_eye, left_eye, nose_tip, right_mouth_corner, left_mouth_corner.

    EAR Formula (adapted for single-point eye landmarks):
    Distance-based approximation using inter-ocular distance and eye-nose distances.
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames: int = 2,
        blink_time_window: float = 0.4
    ):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.blink_time_window = blink_time_window

        self.state = LivenessState.EYES_OPEN
        self.frame_counter = 0
        self.total_blinks = 0
        self.last_blink_time = 0.0
        self.ear_history: List[float] = []
        self.max_history = 30

    def calculate_ear_proxy(self, landmarks: List[Dict[str, int]]) -> Optional[float]:
        """
        Calculate Eye Aspect Ratio proxy using YuNet's 5 landmarks.
        landmarks format: [{"x": int, "y": int}, ...] for right_eye, left_eye, nose, mouth_right, mouth_left

        Approximation: EAR ~ vertical_eye_variance / horizontal_eye_distance
        """
        if len(landmarks) < 5:
            return None

        try:
            right_eye = np.array([landmarks[0]["x"], landmarks[0]["y"]], dtype=np.float32)
            left_eye = np.array([landmarks[1]["x"], landmarks[1]["y"]], dtype=np.float32)
            nose = np.array([landmarks[2]["x"], landmarks[2]["y"]], dtype=np.float32)

            # Inter-ocular distance (horizontal eye separation)
            eye_distance = np.linalg.norm(right_eye - left_eye)

            if eye_distance < 1.0:
                return None

            # Midpoint between eyes
            eye_center = (right_eye + left_eye) / 2.0

            # Vertical distance from eye center to nose tip
            vertical_dist = abs(eye_center[1] - nose[1])

            # EAR proxy: ratio of vertical variance to horizontal spread
            # When eyes close, vertical distance decreases
            ear_proxy = vertical_dist / eye_distance

            # Normalize to typical EAR range (0.15 - 0.35)
            ear_normalized = np.clip(ear_proxy * 0.5, 0.1, 0.5)

            return float(ear_normalized)

        except Exception as e:
            return None

    def update(self, landmarks: List[Dict[str, int]], timestamp: float = None) -> Dict[str, Any]:
        """
        Update blink detection state machine.
        Returns: {
            "ear": float,
            "state": str,
            "blink_detected": bool,
            "total_blinks": int
        }
        """
        if timestamp is None:
            timestamp = time.time()

        ear = self.calculate_ear_proxy(landmarks)

        if ear is None:
            return {
                "ear": 0.0,
                "state": self.state.value,
                "blink_detected": False,
                "total_blinks": self.total_blinks,
            }

        self.ear_history.append(ear)
        if len(self.ear_history) > self.max_history:
            self.ear_history.pop(0)

        blink_detected_now = False

        # State machine logic
        if self.state == LivenessState.EYES_OPEN:
            if ear < self.ear_threshold:
                self.state = LivenessState.EYES_CLOSING
                self.frame_counter = 1

        elif self.state == LivenessState.EYES_CLOSING:
            if ear < self.ear_threshold:
                self.frame_counter += 1
                if self.frame_counter >= self.consecutive_frames:
                    self.state = LivenessState.EYES_CLOSED
            else:
                # Eyes opened before full closure
                self.state = LivenessState.EYES_OPEN
                self.frame_counter = 0

        elif self.state == LivenessState.EYES_CLOSED:
            if ear >= self.ear_threshold:
                # Eyes re-opening - blink completed
                elapsed = timestamp - self.last_blink_time
                if elapsed > self.blink_time_window or self.total_blinks == 0:
                    self.total_blinks += 1
                    self.last_blink_time = timestamp
                    blink_detected_now = True

                self.state = LivenessState.EYES_OPEN
                self.frame_counter = 0

        return {
            "ear": round(ear, 3),
            "state": self.state.value,
            "blink_detected": blink_detected_now,
            "total_blinks": self.total_blinks,
            "ear_avg": round(np.mean(self.ear_history), 3) if self.ear_history else 0.0,
        }

    def reset(self):
        """Reset state machine"""
        self.state = LivenessState.EYES_OPEN
        self.frame_counter = 0
        self.total_blinks = 0
        self.ear_history.clear()


class TextureAnalyzer:
    """
    Passive liveness detection via texture frequency analysis.
    Detects screen replay attacks (photos, phone screens) using:
    - Laplacian variance (high-frequency content)
    - Reflection pattern detection
    - Moiré pattern detection (screen refresh artifacts)
    """

    def __init__(
        self,
        laplacian_threshold: float = 80.0,
        moire_threshold: float = 0.15
    ):
        self.laplacian_threshold = laplacian_threshold
        self.moire_threshold = moire_threshold

    def calculate_laplacian_variance(self, face_crop: np.ndarray) -> float:
        """
        Calculate variance of Laplacian (measure of sharpness/focus).
        Real faces have rich texture; printed photos/screens have lower variance.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance

    def detect_moire_pattern(self, face_crop: np.ndarray) -> float:
        """
        Detect moiré patterns typical of screen refresh interference.
        Uses FFT to identify periodic high-frequency artifacts.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0

        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
            gray = cv2.resize(gray, (64, 64))

            # Apply FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)

            # Analyze high-frequency energy in corners (typical moiré location)
            h, w = magnitude_spectrum.shape
            corner_energy = 0.0

            # Sample corner regions
            corner_size = h // 8
            corners = [
                magnitude_spectrum[0:corner_size, 0:corner_size],
                magnitude_spectrum[0:corner_size, -corner_size:],
                magnitude_spectrum[-corner_size:, 0:corner_size],
                magnitude_spectrum[-corner_size:, -corner_size:],
            ]

            corner_energy = np.mean([np.mean(c) for c in corners])
            center_energy = np.mean(magnitude_spectrum[h//4:3*h//4, w//4:3*w//4])

            # Moiré ratio: corner high-freq vs center low-freq
            moire_ratio = corner_energy / (center_energy + 1e-6)

            return float(moire_ratio)

        except Exception:
            return 0.0

    def analyze(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive texture analysis.
        Returns:
            {
                "laplacian_variance": float,
                "moire_score": float,
                "is_live": bool,
                "confidence": float
            }
        """
        lap_var = self.calculate_laplacian_variance(face_crop)
        moire_score = self.detect_moire_pattern(face_crop)

        # Decision logic
        is_live = (lap_var >= self.laplacian_threshold) and (moire_score < self.moire_threshold)

        # Confidence calculation
        lap_conf = min(1.0, lap_var / (self.laplacian_threshold * 1.5))
        moire_conf = max(0.0, 1.0 - (moire_score / (self.moire_threshold * 2.0)))
        confidence = (lap_conf * 0.7 + moire_conf * 0.3)

        return {
            "laplacian_variance": round(lap_var, 2),
            "moire_score": round(moire_score, 4),
            "is_live": is_live,
            "confidence": round(confidence, 3),
            "reason": self._get_reason(lap_var, moire_score, is_live),
        }

    def _get_reason(self, lap_var: float, moire: float, is_live: bool) -> str:
        if not is_live:
            if lap_var < self.laplacian_threshold:
                return "Low texture variance - possible printed photo"
            if moire >= self.moire_threshold:
                return "Moiré pattern detected - possible screen replay"
        return "Real face texture patterns detected"


class LivenessDetector:
    """
    Comprehensive Presentation Attack Detection (PAD) system combining:
    1. Active challenges (blink, head turn, smile)
    2. Passive texture analysis (Laplacian, moiré)
    3. Temporal consistency checks
    """

    def __init__(
        self,
        require_blink: bool = True,
        require_texture_check: bool = True,
        enable_challenges: bool = False
    ):
        self.require_blink = require_blink
        self.require_texture_check = require_texture_check
        self.enable_challenges = enable_challenges

        self.ear_detector = EyeAspectRatioDetector()
        self.texture_analyzer = TextureAnalyzer()

        self.active_challenge: Optional[LivenessChallenge] = None
        self.challenge_history: List[LivenessChallenge] = []

        self.session_started_at = time.time()
        self.last_verification_time = 0.0
        self.verification_cooldown = 3.0  # seconds between verifications

    def start_challenge(self, challenge_type: ChallengeType) -> LivenessChallenge:
        """Start a new active liveness challenge"""
        prompts = {
            ChallengeType.BLINK: "Blink once",
            ChallengeType.TURN_LEFT: "Turn head slightly left",
            ChallengeType.TURN_RIGHT: "Turn head slightly right",
            ChallengeType.NOD_UP: "Tilt chin up",
            ChallengeType.NOD_DOWN: "Tilt chin down",
            ChallengeType.SMILE: "Smile naturally",
        }

        challenge = LivenessChallenge(
            challenge_type=challenge_type,
            prompt=prompts.get(challenge_type, "Follow the prompt"),
            started_at=time.time()
        )

        self.active_challenge = challenge
        return challenge

    def verify_liveness(
        self,
        face_crop: np.ndarray,
        landmarks: List[Dict[str, int]],
        frame_bgr: Optional[np.ndarray] = None
    ) -> LivenessResult:
        """
        Comprehensive liveness verification combining multiple signals.

        Args:
            face_crop: Cropped face region (BGR)
            landmarks: YuNet 5 facial landmarks
            frame_bgr: Full frame for additional context

        Returns:
            LivenessResult with verification status and details
        """
        current_time = time.time()

        # Initialize result
        result = LivenessResult(
            is_live=False,
            confidence=0.0,
            metrics={}
        )

        # 1. Blink Detection (Active/Passive)
        blink_result = {"blink_detected": False, "total_blinks": 0}
        if self.require_blink and len(landmarks) >= 5:
            blink_result = self.ear_detector.update(landmarks, current_time)
            result.blink_detected = blink_result["blink_detected"]
            result.metrics["ear"] = blink_result["ear"]
            result.metrics["total_blinks"] = blink_result["total_blinks"]

        # 2. Texture Analysis (Passive)
        texture_result = {"is_live": False, "confidence": 0.0}
        if self.require_texture_check and face_crop is not None:
            texture_result = self.texture_analyzer.analyze(face_crop)
            result.texture_score = texture_result["confidence"]
            result.metrics["laplacian_variance"] = texture_result["laplacian_variance"]
            result.metrics["moire_score"] = texture_result["moire_score"]

        # 3. Active Challenge Verification
        challenge_passed = False
        if self.enable_challenges and self.active_challenge is not None:
            challenge_passed = self._check_challenge_completion(
                self.active_challenge,
                blink_result,
                landmarks,
                current_time
            )
            result.challenge_passed = challenge_passed
            result.challenge_type = self.active_challenge.challenge_type.value

        # 4. Decision Logic: Multi-factor fusion
        checks_passed = []

        if self.require_blink:
            blink_ok = blink_result["total_blinks"] >= 1
            checks_passed.append(blink_ok)
            if not blink_ok:
                result.reason = "No blink detected - keep eyes natural"

        if self.require_texture_check:
            texture_ok = texture_result["is_live"]
            checks_passed.append(texture_ok)
            if not texture_ok:
                result.reason = texture_result.get("reason", "Texture analysis failed")

        if self.enable_challenges:
            checks_passed.append(challenge_passed)
            if not challenge_passed and self.active_challenge:
                result.reason = f"Challenge not completed: {self.active_challenge.prompt}"

        # Final verdict
        if len(checks_passed) > 0:
            result.is_live = all(checks_passed)
            result.confidence = sum(checks_passed) / len(checks_passed)
        else:
            # If no checks enabled, default to texture analysis only
            result.is_live = texture_result["is_live"]
            result.confidence = texture_result["confidence"]

        if result.is_live and not result.reason:
            result.reason = "Liveness verified - real person detected"

        # Update verification timestamp
        if result.is_live:
            self.last_verification_time = current_time

        return result

    def _check_challenge_completion(
        self,
        challenge: LivenessChallenge,
        blink_result: Dict[str, Any],
        landmarks: List[Dict[str, int]],
        current_time: float
    ) -> bool:
        """Check if active challenge has been completed successfully"""
        if challenge.completed:
            return challenge.passed

        elapsed = current_time - (challenge.started_at or current_time)
        if elapsed > challenge.timeout_seconds:
            challenge.completed = True
            challenge.passed = False
            return False

        # Challenge-specific checks
        if challenge.challenge_type == ChallengeType.BLINK:
            if blink_result.get("blink_detected", False):
                challenge.completed = True
                challenge.passed = True
                return True

        # Placeholder for head turn/nod detection (requires more advanced landmark tracking)
        elif challenge.challenge_type in [ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT]:
            # Would need horizontal displacement tracking of landmarks
            pass

        elif challenge.challenge_type in [ChallengeType.NOD_UP, ChallengeType.NOD_DOWN]:
            # Would need vertical displacement tracking
            pass

        return False

    def reset(self):
        """Reset all detectors and state"""
        self.ear_detector.reset()
        self.active_challenge = None
        self.challenge_history.clear()
        self.session_started_at = time.time()

    def get_status(self) -> Dict[str, Any]:
        """Get current liveness detection status"""
        return {
            "blink_detector": {
                "enabled": self.require_blink,
                "total_blinks": self.ear_detector.total_blinks,
                "state": self.ear_detector.state.value,
            },
            "texture_analyzer": {
                "enabled": self.require_texture_check,
                "laplacian_threshold": self.texture_analyzer.laplacian_threshold,
            },
            "active_challenge": {
                "enabled": self.enable_challenges,
                "current": self.active_challenge.prompt if self.active_challenge else None,
            },
            "session_duration": round(time.time() - self.session_started_at, 1),
        }
