"""Unit tests for BdSLUserInstructionGenerator."""

import pytest
from core_engine.nlp.user_instruction_generator import BdSLUserInstructionGenerator


class TestBdSLUserInstructionGenerator:
    @pytest.fixture
    def generator(self):
        return BdSLUserInstructionGenerator()

    def test_generate_sign_guide_dhonnobad(self, generator):
        guide = generator.generate_sign_guide("ধন্যবাদ")
        assert guide["status"] == "success"
        assert guide["gloss"] == "ধন্যবাদ"
        assert "posture_steps" in guide
        assert len(guide["posture_steps"]) == 3
        assert "timing_profile" in guide
        assert "facial_cues" in guide
        assert guide["handedness"] in ["এক হাত (Single-Hand)", "দুই হাত (Dual-Hand)"]

    def test_generate_sign_guide_english_or_slug(self, generator):
        guide = generator.generate_sign_guide("dhonnobad")
        assert guide["status"] == "success"
        assert guide["gloss"] == "ধন্যবাদ"

    def test_generate_sign_guide_not_found(self, generator):
        guide = generator.generate_sign_guide("অপরিচিত_শব্দ_xyz")
        assert guide["status"] == "not_found"
        assert "message" in guide

    def test_generate_sentence_guide(self, generator):
        sentence = "ধন্যবাদ ডাক্তার"
        glosses = ["ধন্যবাদ", "ডাক্তার"]
        guide = generator.generate_sentence_guide(sentence, glosses)

        assert guide["sentence"] == sentence
        assert guide["total_steps"] == 2
        assert len(guide["step_by_step_coaching"]) == 2
        assert guide["step_by_step_coaching"][0]["step_no"] == 1
        assert guide["step_by_step_coaching"][0]["gloss"] == "ধন্যবাদ"
        assert guide["step_by_step_coaching"][1]["step_no"] == 2
        assert guide["step_by_step_coaching"][1]["gloss"] == "ডাক্তার"
        assert "general_advice" in guide
