"""Procedural Hyper-Kinematic Synthesizer for BdSL v3 Parametric Specifications.

Generates mathematical 3D skeletal landmark sequences, 21-joint hand poses,
and facial action unit (FACS) blendshapes directly from BdSL v3 DSL schemas.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class HyperKinematicSynthesizer:
    """Mathematical 3D landmark frame generator driven by BdSL Parametric DSL schemas."""

    def __init__(self, fps: int = 60):
        self.fps = fps

    def _cubic_bezier(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Evaluates cubic Bézier curve at parameter t in [0, 1]."""
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3
        )

    def generate_trajectory_frames(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates frame-by-frame 3D skeletal landmarks from a BdSL v3 specification.

        Args:
            schema: BdSL v3 dictionary or validated schema

        Returns:
            List of frame dictionaries containing timestamp, wrist position, 21 hand points, and FACS.
        """
        durations = schema.get("temporal_phases_ms", {"total_ms": 950})
        total_ms = durations.get("total_ms", 950)
        total_frames = max(2, int((total_ms / 1000.0) * self.fps))

        # Node position anchors from kinematics schema or anatomical defaults
        kinematics = schema.get("kinematics", {})
        start_offset = kinematics.get("start_anchor", {}).get("offset_cm", [0.0, 2.0, 4.0])
        end_offset = kinematics.get("end_anchor", {}).get("offset_cm", [0.0, 0.0, 25.0])

        # Normalize cm offsets to 3D normalized spatial coordinates
        p_start = np.array([
            0.5 + start_offset[0] * 0.01,
            0.45 + start_offset[1] * 0.01,
            -0.1 + start_offset[2] * 0.01
        ], dtype=np.float32)

        p_end = np.array([
            0.5 + end_offset[0] * 0.01,
            0.60 + end_offset[1] * 0.01,
            -0.1 - end_offset[2] * 0.01
        ], dtype=np.float32)

        # Control points creating natural human reaching arc
        p_ctrl1 = (p_start * 0.7 + p_end * 0.3) + np.array([0.0, 0.03, -0.05], dtype=np.float32)
        p_ctrl2 = (p_start * 0.3 + p_end * 0.7) + np.array([0.0, 0.05, -0.05], dtype=np.float32)

        handshape_code = schema.get("phonetics", {}).get("handshape_code", "HS_FLAT_BENT_THUMB")
        facs_schema = schema.get("facial_action_units", {})
        au12_target = facs_schema.get("AU12_lip_corner_puller", 0.8)
        head_pitch_target = facs_schema.get("head_pose", {}).get("pitch", -4.0)

        frames = []
        for f in range(total_frames):
            t = f / float(total_frames - 1)
            # Smooth cubic ease-in-out curve
            t_eased = 3 * t ** 2 - 2 * t ** 3
            wrist_pos = self._cubic_bezier(p_start, p_ctrl1, p_ctrl2, p_end, t_eased)

            # Synthesize 21 articulation joints for hand
            right_hand_points = self._synthesize_hand_pose(wrist_pos, handshape_code, t_eased)

            # Dynamic FACS Action Unit interpolation
            au_data = {
                "AU12": float(round(au12_target * min(1.0, t * 2.0), 3)),
                "head_pitch": float(round(head_pitch_target * np.sin(t * np.pi), 3))
            }

            frames.append({
                "frame_idx": f,
                "timestamp_ms": int((f / self.fps) * 1000),
                "right_wrist": [float(round(v, 4)) for v in wrist_pos.tolist()],
                "right_hand": right_hand_points,
                "facs": au_data
            })

        return frames

    def _synthesize_hand_pose(
        self,
        wrist: np.ndarray,
        handshape: str,
        progress: float
    ) -> List[List[float]]:
        """Calculates 21 finger joint coordinates based on handshape code and progress."""
        joints = []
        # Wrist root joint
        joints.append([float(round(v, 4)) for v in wrist.tolist()])

        # 5 fingers with 4 joints each (Thumb, Index, Middle, Ring, Pinky)
        for finger_idx in range(5):
            for joint_depth in range(1, 5):
                # Spread angle and knuckle spacing
                spread_x = (finger_idx - 2) * 0.015
                depth_y = -joint_depth * 0.02

                if handshape == "HS_FLAT_BENT_THUMB":
                    # Flat palm with bent thumb
                    if finger_idx == 0:  # Thumb
                        offset_z = joint_depth * 0.008
                        spread_x = -0.02 + joint_depth * 0.004
                    else:
                        offset_z = joint_depth * 0.002
                elif handshape == "HS_INDEX_EXTENDED":
                    # Pointing index
                    if finger_idx == 1:
                        offset_z = 0.0
                    else:
                        depth_y = -0.015  # Curled fist
                        offset_z = joint_depth * 0.01
                elif handshape == "HS_FIST":
                    depth_y = -0.015
                    offset_z = joint_depth * 0.012
                else:
                    offset_z = 0.0

                joint_pt = wrist + np.array([spread_x, depth_y, offset_z], dtype=np.float32)
                joints.append([float(round(v, 4)) for v in joint_pt.tolist()])

        return joints[:21]


class MultiSignSequenceBlender:
    """
    Blends multiple BdSL v3 sign specifications into a continuous natural motion stream
    using co-articulation and Hermite/slerp interpolation.
    """

    def __init__(self, fps: int = 60, transition_ms: int = 150):
        self.fps = fps
        self.transition_ms = transition_ms
        self.transition_frames = max(2, int((transition_ms / 1000.0) * fps))
        self.synthesizer = HyperKinematicSynthesizer(fps=fps)

    def _hermite_interpolate(
        self, p0: np.ndarray, v0: np.ndarray, p1: np.ndarray, v1: np.ndarray, t: float
    ) -> np.ndarray:
        """Cubic Hermite interpolation maintaining smooth acceleration between two points."""
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1

    def generate_transition_frames(
        self, last_frame: Dict[str, Any], next_frame: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generates transition frames between the end of one sign and the start of the next."""
        p_start = np.array(last_frame["right_wrist"], dtype=np.float32)
        p_end = np.array(next_frame["right_wrist"], dtype=np.float32)

        v_start = (p_end - p_start) * 0.25
        v_end = (p_end - p_start) * 0.25

        hand_start = np.array(last_frame["right_hand"], dtype=np.float32)
        hand_end = np.array(next_frame["right_hand"], dtype=np.float32)

        facs_start = last_frame.get("facs", {"AU12": 0.0, "head_pitch": 0.0})
        facs_end = next_frame.get("facs", {"AU12": 0.0, "head_pitch": 0.0})

        transition = []
        for i in range(self.transition_frames):
            t = (i + 1) / (self.transition_frames + 1)
            t_smooth = 3 * t**2 - 2 * t**3

            interp_wrist = self._hermite_interpolate(p_start, v_start, p_end, v_end, t_smooth)
            interp_hand = (1 - t_smooth) * hand_start + t_smooth * hand_end

            interp_facs = {
                "AU12": float(round((1 - t_smooth) * facs_start.get("AU12", 0.0) + t_smooth * facs_end.get("AU12", 0.0), 3)),
                "head_pitch": float(round((1 - t_smooth) * facs_start.get("head_pitch", 0.0) + t_smooth * facs_end.get("head_pitch", 0.0), 3))
            }

            transition.append({
                "right_wrist": [float(round(v, 4)) for v in interp_wrist.tolist()],
                "right_hand": [[float(round(c, 4)) for c in pt] for pt in interp_hand.tolist()],
                "facs": interp_facs,
                "is_transition": True
            })

        return transition

    def blend_sentence_stream(self, sign_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Blends a sequence of BdSL v3 sign specs into a single continuous stream of frames."""
        if not sign_specs:
            return []

        full_stream = []
        for spec in sign_specs:
            sign_frames = self.synthesizer.generate_trajectory_frames(spec)
            if not sign_frames:
                continue

            if full_stream:
                last_frame = full_stream[-1]
                first_frame = sign_frames[0]
                transitions = self.generate_transition_frames(last_frame, first_frame)
                full_stream.extend(transitions)

            for frame in sign_frames:
                frame_copy = dict(frame)
                frame_copy["is_transition"] = False
                full_stream.append(frame_copy)

        for idx, frame in enumerate(full_stream):
            frame["frame_idx"] = idx
            frame["timestamp_ms"] = int((idx / self.fps) * 1000)

        return full_stream
