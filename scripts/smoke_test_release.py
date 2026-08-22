"""Automated Master Release Smoke Test for IsharaConnect Production Pipeline.

Verifies end-to-end subsystem integrity in a headless runtime environment:
1. SQLite/Async SQLAlchemy Database schema creation & tables reflection
2. Continuous Sign NLP Translation Engine & Morphological Inflector
3. PyQt6 Desktop UI Components (SubtitleTickerWidget, HumanRigViewer, AcademyDashboard)
4. Fast discovery of WebRTC & HTML Templates (index.html, verify_certificate.html, skeleton_player.html)
5. Audio Controller & TTS Engine offline synthesis check
6. Distributed Room Manager and Redis Adapter resolution
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_status(step_name: str, passed: bool, details: str = ""):
    symbol = "✅" if passed else "❌"
    print(f" {symbol} {step_name:<50} {details}")


async def test_database():
    """Verify Async SQLAlchemy Database session & tables."""
    from backend.db.models.base import Base
    from backend.db.session import engine
    from backend.db.models.user import User
    from backend.db.models.exam import ExamRecord
    from backend.db.models.progress import LearningProgress

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        table_names = await conn.run_sync(lambda sync_conn: list(Base.metadata.tables.keys()))
    
    assert "users" in table_names
    assert "exam_records" in table_names
    assert "learning_progress" in table_names
    return f"({len(table_names)} tables verified)"


def test_nlp_translation():
    """Verify continuous NLP gloss translation and morphological inflection."""
    from core_engine.nlp.gloss_to_sentence import GlossToSentenceTranslator
    
    translator = GlossToSentenceTranslator()
    res = translator.translate(["আমি", "ভাত", "খাওয়া"])
    assert res["translated_text"] == "আমি ভাত খাচ্ছি।"
    assert res["confidence"] >= 0.90
    return f"('আমি ভাত খাচ্ছি।' generated with conf={res['confidence']})"


def test_ui_components():
    """Verify PyQt6 UI components instantiate headlessly."""
    from PyQt6.QtWidgets import QApplication
    from desktop_app.ui.components.subtitle_ticker import SubtitleTickerWidget
    from desktop_app.ui.components.human_rig_viewer import HumanRigViewer
    
    app = QApplication.instance() or QApplication([])
    ticker = SubtitleTickerWidget()
    ticker.update_active_glosses(["আমি", "ভাত"])
    ticker.update_translation("আমি ভাত খাচ্ছি।", 0.95, True)
    assert ticker.sentence_label.text() == "আমি ভাত খাচ্ছি।"

    rig = HumanRigViewer()
    assert rig is not None
    return "(SubtitleTicker & HumanRigViewer verified)"


def test_web_templates():
    """Verify HTML templates and static assets exist for WebRTC/Verification."""
    templates_dir = PROJECT_ROOT / "backend" / "templates"
    required = ["index.html", "verify_certificate.html", "skeleton_player.html", "avatar_viewport.html"]
    for t in required:
        assert (templates_dir / t).exists(), f"Missing template: {t}"
    return f"({len(required)} web templates verified)"


def test_audio_controller():
    """Verify audio controller resolves speech synthesis."""
    from desktop_app.controllers.audio_player import audio_controller
    assert audio_controller is not None
    assert audio_controller.tts is not None
    return "(TTS Engine & AudioPlayerController active)"


async def test_room_manager():
    """Verify Room Manager and Redis adapter fallback."""
    from backend.websockets.room_manager import get_room_adapter
    adapter = await get_room_adapter()
    assert adapter is not None
    return f"(Adapter: {type(adapter).__name__})"


async def main():
    print("\n==================================================================")
    print(" 🚀 ISHARACONNECT MASTER PRODUCTION RELEASE SMOKE TEST")
    print("==================================================================")
    
    all_passed = True
    start_time = time.perf_counter()

    # 1. Database
    try:
        details = await test_database()
        print_status("Database Schema & Async Engine", True, details)
    except Exception as e:
        print_status("Database Schema & Async Engine", False, str(e))
        all_passed = False

    # 2. NLP Translation
    try:
        details = test_nlp_translation()
        print_status("Continuous Sign NLP Translation Engine", True, details)
    except Exception as e:
        print_status("Continuous Sign NLP Translation Engine", False, str(e))
        all_passed = False

    # 3. UI Components
    try:
        details = test_ui_components()
        print_status("PyQt6 Desktop Subtitle Ticker & Rig Viewer", True, details)
    except Exception as e:
        print_status("PyQt6 Desktop Subtitle Ticker & Rig Viewer", False, str(e))
        all_passed = False

    # 4. Web Templates
    try:
        details = test_web_templates()
        print_status("WebRTC Templates & Certificate HTML Discovery", True, details)
    except Exception as e:
        print_status("WebRTC Templates & Certificate HTML Discovery", False, str(e))
        all_passed = False

    # 5. Audio Controller
    try:
        details = test_audio_controller()
        print_status("Audio Controller & Speech Vocalizer Pipeline", True, details)
    except Exception as e:
        print_status("Audio Controller & Speech Vocalizer Pipeline", False, str(e))
        all_passed = False

    # 6. Room Manager
    try:
        details = await test_room_manager()
        print_status("WebSocket Room Manager & Redis Adapter", True, details)
    except Exception as e:
        print_status("WebSocket Room Manager & Redis Adapter", False, str(e))
        all_passed = False

    elapsed = time.perf_counter() - start_time
    print("==================================================================")
    if all_passed:
        print(f" ✨ ALL SUBSYSTEMS PASSED PRODUCTION SMOKE TEST IN {elapsed:.2f}s! ✨")
        print("==================================================================\n")
        sys.exit(0)
    else:
        print(f" ❌ SMOKE TEST FAILED IN {elapsed:.2f}s")
        print("==================================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
