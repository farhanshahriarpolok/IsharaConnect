"""Antigravity Agent Pipeline for BdSL Reasoning, DSL Generation & Landmark Validation."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core_engine.dsl.bdsl_tools import get_sign_dsl_tool, get_sign_landmarks_tool
from core_engine.nlp.bengali_inflection import BengaliMorphologicalInflector


@dataclass
class Task:
    """Represents an agent execution task."""
    description: str
    expected_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """Autonomous reasoning agent for BdSL translation and DSL synthesis."""

    def __init__(
        self,
        name: str,
        role: str,
        instructions: str,
        tools: Optional[List[Callable]] = None
    ):
        self.name = name
        self.role = role
        self.instructions = instructions
        self.tools = tools or []
        self.inflector = BengaliMorphologicalInflector()

    async def execute(self, task: Task) -> Dict[str, Any]:
        """
        Executes BdSL translation task:
        1. Extracts glosses from Bengali input sentence.
        2. Queries `get_sign_dsl_tool` for each token.
        3. Queries `get_sign_landmarks_tool` for skeletal animation verification.
        4. Compiles structured JSON response.
        """
        desc = task.description

        # Extract text from description (e.g., "'আমি ভাত খাচ্ছি।'" or direct sentence)
        match = re.search(r"'(.*?)'|\"(.*?)\"", desc)
        if match:
            raw_text = match.group(1) or match.group(2)
        else:
            raw_text = desc

        # Normalize and tokenize Bengali text into glosses
        clean_text = raw_text.strip()
        tokens = [t.strip("।,?! ") for t in clean_text.split() if t.strip("।,?! ")]

        gloss_sequence = []
        dsl_sequence = []
        landmark_payloads = []

        # Common word-to-root gloss map
        root_map = {
            "খাচ্ছি": "খাওয়া",
            "খাব": "খাওয়া",
            "খেয়েছি": "খাওয়া",
            "খায়": "খাওয়া",
            "যাচ্ছি": "যাওয়া",
            "যাব": "যাওয়া",
            "গেছি": "যাওয়া",
            "আছি": "থাকা",
            "আছেন": "কেমন আছেন",
            "ভাত": "ভাত",
            "আমি": "আমি",
            "ধন্যবাদ": "ধন্যবাদ",
            "স্বাগতম": "স্বাগতম",
            "সাহায্য": "সাহায্য",
            "ভালো": "ভালো",
        }

        for token in tokens:
            gloss = root_map.get(token, token)
            gloss_sequence.append(gloss)

            # 1. Fetch Parametric DSL
            dsl_res = get_sign_dsl_tool(gloss)
            if dsl_res.get("status") == "success":
                dsl_data = dsl_res.get("data", {})
                dsl_sequence.append({
                    "gloss": gloss,
                    "slug": dsl_data.get("slug", gloss),
                    "dsl": dsl_data.get("dsl", {}),
                    "handedness": dsl_data.get("handedness", "single"),
                    "motion_type": dsl_data.get("motion_type", "dynamic"),
                })

                # 2. Check Landmark Availability
                slug = dsl_data.get("slug", gloss)
                lm_res = get_sign_landmarks_tool(slug)
                landmark_payloads.append({
                    "gloss": gloss,
                    "slug": slug,
                    "available": lm_res.get("status") == "success",
                    "status": lm_res.get("status")
                })
            else:
                dsl_sequence.append({
                    "gloss": gloss,
                    "slug": None,
                    "status": "not_in_dictionary"
                })
                landmark_payloads.append({
                    "gloss": gloss,
                    "slug": None,
                    "available": False,
                    "status": "not_found"
                })

        return {
            "status": "success",
            "agent": self.name,
            "input_text": raw_text,
            "gloss_sequence": gloss_sequence,
            "dsl_sequence": dsl_sequence,
            "landmarks_status": landmark_payloads,
            "all_landmarks_available": all(item.get("available", False) for item in landmark_payloads)
        }


# 1. Instantiate default IsharaConnect Agent
ishara_agent = Agent(
    name="IsharaConnect-Agent",
    role="Sign Language Expert & Translator",
    instructions="""
    You are an expert BdSL (Bangla Sign Language) reasoning agent.
    When given a Bengali sentence or word:
    1. Extract the core gloss sequence.
    2. Use `get_sign_dsl_tool` to retrieve the Parametric DSL notations.
    3. Use `get_sign_landmarks_tool` when skeletal animation data is needed.
    4. Provide the exact animation sequence and execution plan.
    """,
    tools=[get_sign_dsl_tool, get_sign_landmarks_tool]
)


# 2. Pipeline Execution helper
async def run_pipeline(input_text: str) -> Dict[str, Any]:
    """Runs end-to-end BdSL translation and landmark verification pipeline."""
    task = Task(
        description=f"Translate this Bengali sentence into BdSL DSL sequence and verify landmark availability: '{input_text}'",
        expected_output="A structured JSON response with gloss, DSL tokens, and landmark payload status."
    )
    result = await ishara_agent.execute(task)
    return result
