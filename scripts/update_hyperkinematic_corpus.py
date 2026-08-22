"""Updates master_bdsl_lexicon.json to v3.1.0 with HyperKinematic corpus."""

import json
from pathlib import Path

p = Path("dataset/lexicon/master_bdsl_lexicon.json")
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

# Update metadata
data["version"] = "3.1.0"
data["dataset_metadata"] = {
    "name": "IsharaConnect-HyperKinematic-Master-Corpus",
    "version": "3.1.0",
    "release_year": 2026,
    "fps": 60,
    "coordinate_system": "Normalized_3D_RightHanded",
    "tiers_covered": ["Tier_1_Dactylology", "Tier_2_Lexicon", "Tier_3_CSLR", "Tier_4_SLT", "Tier_5_HyperKinematics"]
}

# Add tier 1 dactylology
data["tier_1_complete_dactylology"] = {
    "VOWELS": {
        "অ": { "handshape": "H_O_SHAPE", "loc": "L_CHEST", "ori": "O_OUT", "mov": "M_STILL", "trigger": "T0_DEFAULT_KAR", "user_guide": "Form an open circular 'O' shape with thumb and fingers touching at tips facing camera." },
        "আ": { "handshape": "H_PALM_OPEN_THUMB_OUT", "loc": "L_CHEST", "ori": "O_OUT", "mov": "M_SLIGHT_UP", "trigger": "T1_AA_KAR", "user_guide": "Extend 4 fingers up, thumb 90° out; elevate forearm 3 cm vertically." },
        "ই": { "handshape": "H_INDEX_CURVED", "loc": "L_CHIN", "ori": "O_IN", "mov": "M_STILL", "trigger": "T1_I_KAR", "user_guide": "Hook index finger into arc pointing to right chin line." },
        "ঈ": { "handshape": "H_INDEX_CURVED_EXT", "loc": "L_CHIN", "ori": "O_IN", "mov": "M_PULL_RIGHT", "trigger": "T2_EE_KAR", "user_guide": "Hooked index finger gliding 8 cm horizontally right across chin." },
        "উ": { "handshape": "H_V_BENT", "loc": "L_CHEST", "ori": "O_DOWN", "mov": "M_PUSH_DOWN", "trigger": "T1_U_KAR", "user_guide": "Inverted 'V' index/middle fingers pushing 5 cm downward." },
        "ঊ": { "handshape": "H_V_BENT_DOUBLE", "loc": "L_CHEST", "ori": "O_DOWN", "mov": "M_PULL_DOWN", "trigger": "T2_OO_KAR", "user_guide": "Inverted 'V' descending in a 2-step pulse (2cm + 2cm)." },
        "ঋ": { "handshape": "H_THREE_FINGERS_CURL", "loc": "L_CHEST", "ori": "O_OUT", "mov": "M_STILL", "trigger": "T3_RRI_KAR", "user_guide": "Three clawed fingers directed forward at chest level." },
        "এ": { "handshape": "H_TRIANGLE_INDEX_THUMB", "loc": "L_CHEST", "ori": "O_OUT", "mov": "M_STILL", "trigger": "T1_E_KAR", "user_guide": "Triangular aperture between index and thumb held steadily." },
        "ঐ": { "handshape": "H_INDEX_MIDDLE_CROSS", "loc": "L_CHEST", "ori": "O_OUT", "mov": "M_WOBBLE", "trigger": "T2_OI_KAR", "user_guide": "Middle finger crossed behind index with gentle lateral wave." },
        "ও": { "handshape": "H_O_CIRCULAR", "loc": "L_NEUTRAL", "ori": "O_OUT", "mov": "M_STILL", "trigger": "T1_O_KAR", "user_guide": "All fingertips meeting thumb tip in compact circular aperture." },
        "ঔ": { "handshape": "H_O_PLUS_HOOK", "loc": "L_NEUTRAL", "ori": "O_OUT", "mov": "M_ARC_UP", "trigger": "T2_OU_KAR", "user_guide": "'O' shape transitioning into an upward flicking hook vector." }
    },
    "CONJUNCTS_AND_SPECIAL": {
        "ক্ষ": { "base": "ক", "trigger": "T4", "components": ["ক", "ষ"], "flow": ["H_INDEX_BENT", "H_THREE_FINGERS_CURV"], "mov": "M_CROSS" },
        "জ্ঞ": { "base": "গ", "trigger": "T5", "components": ["জ", "ঞ"], "flow": ["H_GUN_SHAPE_DOWN", "H_THUMB_PINKY_TOUCH"], "mov": "M_DOUBLE_TAP" }
    }
}

# Add continuous corpus
data["tier_3_4_continuous_corpus"] = [
    {
        "sentence_id": "BDSL_SEN_001",
        "spoken_bengali": "ভূমিকম্প হলে লিফট দিয়ে নামবেন না, সিঁড়ি দিয়ে সাবধানে নামুন।",
        "syntactic_glosses": ["ভূমিকম্প", "ঘটলে", "লিফট", "উঠানামা", "না", "সিঁড়ি", "সাবধান", "নামা"],
        "rules": ["CONJUNCTION_DROPPING", "COMPOUND_UNPACKING", "TERMINAL_NEGATION", "CONDITIONAL_ANTECEDENT"],
        "nmm_facs": { "antecedent": { "AU01": 0.85, "AU02": 0.80, "pitch": 6.0 }, "negation": { "AU04": 0.60, "AU15": 0.70, "yaw_osc": 18.0 } },
        "total_duration_ms": 6450
    },
    {
        "sentence_id": "BDSL_SEN_002",
        "spoken_bengali": "জরুরি সাহায্যের জন্য দ্রুত ডাক্তার এবং অ্যাম্বুলেন্স ডাকুন!",
        "syntactic_glosses": ["জরুরি", "সাহায্য", "ডাক্তার", "হাসপাতাল-গাড়ি", "ডাকুন", "তাড়াতাড়ি"],
        "rules": ["COMPOUND_TRANSLATION", "TERMINAL_VELOCITY_BOOSTER"],
        "nmm_facs": { "urgency": { "AU01": 0.90, "AU02": 0.90, "AU25": 0.60, "pitch": 5.0 } },
        "total_duration_ms": 4820
    }
]

# Ensure new signs exist in signs list
existing_slugs = {s.get("slug") for s in data.get("signs", [])}
new_signs = [
    {
        "slug": "cha",
        "sign_id": "BDSL_V3_00105",
        "label_bn": "চা",
        "label_en": "Tea",
        "category": "Food & Daily Life",
        "handedness": "dual",
        "handshape": "HS_PINCH_CUP",
        "stokoe_notation": "⫸𝄆√",
        "bezier_anchors_3d": {"P0": [0.0, 0.50, 0.20], "P1": [0.0, 0.45, 0.18], "P2": [0.0, 0.42, 0.15], "P3": [0.0, 0.45, 0.18]},
        "facs_action_units": {"AU12": 0.30, "head_pitch": -2.0},
        "contact_physics": {"plane": "CORONAL", "body_anchor": "CHEST_MID", "touch_type": "VERTICAL_TEABAG_DIP"},
        "timing_ms": {"prep": 140, "stroke": 450, "retract": 150, "total": 740},
        "target_body_anchor": "CHEST_MID",
        "articulator_type": "THUMB_INDEX_PINCH",
        "motion_type": "TAP_TWICE",
        "user_guide": "Hold left hand as a cup, dip right pinched fingers vertically into the cup twice like a teabag."
    },
    {
        "slug": "coffee",
        "sign_id": "BDSL_V3_00106",
        "label_bn": "কফি",
        "label_en": "Coffee",
        "category": "Food & Daily Life",
        "handedness": "dual",
        "handshape": "HS_FIST_ON_FIST",
        "stokoe_notation": "⫸⫷☍",
        "bezier_anchors_3d": {"P0": [0.0, 0.50, 0.20], "P1": [0.05, 0.50, 0.20], "P2": [0.0, 0.50, 0.25], "P3": [-0.05, 0.50, 0.20]},
        "facs_action_units": {"AU12": 0.20, "head_pitch": 0.0},
        "contact_physics": {"plane": "TRANSVERSE", "body_anchor": "CHEST_MID", "touch_type": "CIRCULAR_GRINDING_ORBIT"},
        "timing_ms": {"prep": 160, "stroke": 550, "retract": 150, "total": 860},
        "target_body_anchor": "CHEST_MID",
        "articulator_type": "PALM_CLASP",
        "motion_type": "CIRCULAR_ORBIT",
        "user_guide": "Place right fist over left fist, rotate right fist in a horizontal grinding circle twice."
    },
    {
        "slug": "dudh",
        "sign_id": "BDSL_V3_00108",
        "label_bn": "দুধ",
        "label_en": "Milk",
        "category": "Food & Daily Life",
        "handedness": "dual",
        "handshape": "HS_SQUEEZE_FIST",
        "stokoe_notation": "⫸⫷∿",
        "bezier_anchors_3d": {"P0": [-0.15, 0.50, 0.20], "P1": [0.15, 0.50, 0.20], "P2": [-0.15, 0.58, 0.20], "P3": [0.15, 0.58, 0.20]},
        "facs_action_units": {"AU00": 0.0, "head_pitch": 0.0},
        "contact_physics": {"plane": "SAGITTAL", "body_anchor": "CHEST_MID", "touch_type": "ALTERNATING_PULL_DOWN"},
        "timing_ms": {"prep": 150, "stroke": 600, "retract": 150, "total": 900},
        "target_body_anchor": "CHEST_MID",
        "articulator_type": "FINGERTIPS_FLAT",
        "motion_type": "PULL_DOWN",
        "user_guide": "Squeeze both fists alternately while pulling downward in chest space imitating milking."
    }
]

for ns in new_signs:
    if ns["slug"] not in existing_slugs:
        data["signs"].append(ns)
        existing_slugs.add(ns["slug"])

data["total_signs"] = len(data["signs"])

with open(p, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Successfully updated master_bdsl_lexicon.json to v3.1.0 with {len(data['signs'])} signs.")
