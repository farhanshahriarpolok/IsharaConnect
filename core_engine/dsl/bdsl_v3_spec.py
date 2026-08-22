"""BdSL v3 Parametric Kinematics, Phonetics & FACS Morphosyntax Specification Engine.

Models high-fidelity sign language synthesis parameters:
- Phonetics: Handshape codes, Stokoe notation, dominant/subordinate hands
- Kinematics: Bézier spline trajectories, spatial anchors, Euler joint rotations, velocity profiles
- Facial Action Units (FACS): AU blendshapes, head pose, gaze vectors
- Contact Physics: Body surface contact, force normalization, collision phases
- Temporal Phases: Preparation, stroke, hold, and retraction timing in milliseconds
- Morphosyntax: Part of speech, root lemma, non-manual markers (NMM)
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class BdSLPhonetics(BaseModel):
    handshape_code: str = Field(..., description="Standardized handshape code (e.g. HS_FLAT_BENT_THUMB)")
    stokoe_notation: str = Field(..., description="Stokoe phonological notation")
    primary_dominant_hand: str = Field("right", description="'right' or 'left'")
    two_handed: bool = Field(False, description="Whether sign requires dual hand articulation")


class AnchorPoint(BaseModel):
    body_part: str = Field(..., description="Anatomical anchor landmark (e.g. CHIN, NOSE, MID_CHEST)")
    offset_cm: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="[x, y, z] offset in cm")


class JointEulerRotations(BaseModel):
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    flexion_deg: Optional[float] = None


class VelocityProfile(BaseModel):
    peak_velocity_ms: float = 1.0
    ease_type: str = "CUBIC_OUT"


class BdSLKinematics(BaseModel):
    trajectory_spline: str = Field("BEZIER_P0_P1_P2", description="Trajectory interpolation curve type")
    start_anchor: AnchorPoint
    end_anchor: AnchorPoint
    joint_rotations_euler: Dict[str, Any] = Field(default_factory=dict)
    velocity_profile: VelocityProfile = Field(default_factory=VelocityProfile)


class HeadPose(BaseModel):
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


class BdSLFacialActionUnits(BaseModel):
    AU06_cheek_raiser: float = 0.0
    AU12_lip_corner_puller: float = 0.0
    AU25_lips_part: float = 0.0
    AU01_inner_brow_raiser: float = 0.0
    AU02_outer_brow_raiser: float = 0.0
    AU04_brow_lowerer: float = 0.0
    head_pose: HeadPose = Field(default_factory=HeadPose)
    gaze_vector: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])


class BdSLContactPhysics(BaseModel):
    has_contact: bool = False
    contact_surface: Optional[str] = None
    contact_phase: Optional[str] = None
    contact_force_norm: float = 0.0


class BdSLTemporalPhases(BaseModel):
    preparation_duration: int = 150
    stroke_duration: int = 400
    hold_duration: int = 100
    retraction_duration: int = 200
    total_ms: int = 850


class BdSLMorphosyntax(BaseModel):
    pos: str = Field("NOUN", description="Part of Speech tag (e.g. NOUN, VERB, INTERJECTION)")
    root_lemma: str
    synonyms: List[str] = Field(default_factory=list)
    requires_nmm_negation: bool = False


class BdSLV3SignSpec(BaseModel):
    """Complete BdSL Version 3.0 Parametric Kinematic & Phonetic Specification."""

    sign_id: str
    gloss_bn: str
    gloss_en: str
    phonetics: BdSLPhonetics
    kinematics: BdSLKinematics
    facial_action_units: BdSLFacialActionUnits
    contact_physics: BdSLContactPhysics
    temporal_phases_ms: BdSLTemporalPhases
    morphosyntax: BdSLMorphosyntax

    def get_stokoe_summary(self) -> str:
        """Returns concise linguistic phonetic representation."""
        return f"{self.gloss_bn} [{self.phonetics.stokoe_notation}] - {self.phonetics.handshape_code}"

    def compute_bezier_trajectory(self, num_samples: int = 30) -> List[List[float]]:
        """Computes 3D Bézier interpolation points between start and end anchors."""
        p0 = self.kinematics.start_anchor.offset_cm
        p2 = self.kinematics.end_anchor.offset_cm
        # Control point with parabolic upward arc
        p1 = [
            (p0[0] + p2[0]) / 2.0,
            max(p0[1], p2[1]) + 5.0,
            (p0[2] + p2[2]) / 2.0,
        ]

        trajectory = []
        for i in range(num_samples):
            t = i / float(num_samples - 1) if num_samples > 1 else 0.0
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
            z = (1 - t) ** 2 * p0[2] + 2 * (1 - t) * t * p1[2] + t ** 2 * p2[2]
            trajectory.append([round(x, 2), round(y, 2), round(z, 2)])
        return trajectory
