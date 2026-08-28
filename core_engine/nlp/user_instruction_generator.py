"""BdSL User Instruction & Movement Execution Guide Generator.

Generates human-readable, step-by-step Bengali motion instructions and coaching
guidance from the Master BdSL Lexical & Kinematics database for real-time practice.
"""

from typing import Any, Dict, List, Optional
from core_engine.nlp.master_lexicon import MasterBdSLLexicon


class BdSLUserInstructionGenerator:
    """মাস্টার লেক্সিকন ও কাইনেম্যাটিক্স ডাটাবেস থেকে ব্যবহারকারীর জন্য 
    সহজবোধ্য বাংলায় তাৎক্ষণিক মুভমেন্ট ও এক্সিকিউশন গাইড তৈরি করে।
    """

    def __init__(self, lexicon_engine: Optional[MasterBdSLLexicon] = None):
        self.lexicon_engine = lexicon_engine or MasterBdSLLexicon()
        self.signs_db: Dict[str, Dict[str, Any]] = {}
        for s in self.lexicon_engine.get_all_signs():
            bn = s.get("label_bn", "").strip()
            slug = s.get("slug", "").strip()
            if bn:
                self.signs_db[bn] = s
            if slug:
                self.signs_db[slug] = s

    def generate_sign_guide(self, gloss_bn: str) -> Dict[str, Any]:
        """Generates structured posture, movement, and facial instructions for a single sign."""
        clean_gloss = str(gloss_bn).strip()
        sign_data = self.signs_db.get(clean_gloss) or self.lexicon_engine.get_sign_by_gloss(clean_gloss)
        
        if not sign_data:
            return {
                "status": "not_found",
                "message": f"'{gloss_bn}' শব্দটির নির্দেশিকা ডাটাবেসে পাওয়া যায়নি।"
            }

        timing = sign_data.get("timing_ms", {})
        facs = sign_data.get("facs_action_units", {})
        contact = sign_data.get("contact_physics", {})
        
        # Resolve posture elements
        anchor = (
            sign_data.get("target_body_anchor")
            or contact.get("body_anchor")
            or "বুকে"
        )
        handshape = sign_data.get("handshape") or "নির্দিষ্ট হ্যান্ডশেপ"
        stokoe = sign_data.get("stokoe_notation")
        stokoe_mov = stokoe.get("movement") if isinstance(stokoe, dict) else (str(stokoe) if stokoe else None)
        motion_type = (
            sign_data.get("motion_type")
            or stokoe_mov
            or "নির্দিষ্ট ভঙ্গি"
        )
        user_guide = (
            sign_data.get("user_guide")
            or sign_data.get("description")
            or "প্রদত্ত অ্যানিমেশন অনুযায়ী হাত সঞ্চালন করুন।"
        )
        
        # Facial cues from FACS Action Units
        if facs.get("AU12", 0) > 0.3 or facs.get("AU06", 0) > 0.3:
            facial_cues = "হাসিমুখ বজায় রাখুন"
        elif facs.get("AU01", 0) > 0.4 or facs.get("AU02", 0) > 0.4:
            facial_cues = "ভ্রু উপরে তুলে প্রশ্ন করুন"
        elif facs.get("AU04", 0) > 0.4:
            facial_cues = "মনোযোগ সহকারে ভ্রু সামান্য কুঁচকে রাখুন"
        else:
            facial_cues = "স্বাভাবিক দৃষ্টি রাখুন"

        return {
            "status": "success",
            "gloss": sign_data.get("label_bn", gloss_bn),
            "category": sign_data.get("category", "General"),
            "handedness": "দুই হাত (Dual-Hand)" if sign_data.get("handedness") == "dual" else "এক হাত (Single-Hand)",
            "primary_instruction": user_guide,
            "posture_steps": [
                f"১. শুরুর অবস্থান: হাতকে {anchor} অবস্থানে রাখুন।",
                f"২. হাতের গঠন: {handshape} অনুযায়ী আঙুল সাজান।",
                f"৩. গতিবিধি: {motion_type} অনুযায়ী হাত সঞ্চালন করুন।"
            ],
            "facial_cues": facial_cues,
            "timing_profile": {
                "duration_sec": round(timing.get("total", 600) / 1000.0, 2),
                "stroke_ms": timing.get("stroke", 400)
            }
        }

    def generate_sentence_guide(self, sentence_bn: str, gloss_sequence: List[str]) -> Dict[str, Any]:
        """Generates step-by-step sequential coaching instructions for a full sentence."""
        step_guides = []
        for idx, gloss in enumerate(gloss_sequence):
            guide = self.generate_sign_guide(gloss)
            step_guides.append({
                "step_no": idx + 1,
                "gloss": gloss,
                "action": guide.get("primary_instruction", "সাইনটি সম্পাদন করুন।"),
                "facial_expression": guide.get("facial_cues", "স্বাভাবিক")
            })

        return {
            "sentence": sentence_bn,
            "total_steps": len(gloss_sequence),
            "step_by_step_coaching": step_guides,
            "general_advice": "প্রতিটি শব্দের মাঝে হাত হুট করে নামিয়ে ফেলবেন না; এক ভঙ্গি থেকে পরবর্তী ভঙ্গিতে মসৃণভাবে হাত সরিয়ে নিন।"
        }
