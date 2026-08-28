"""3D Realistic Mannequin Signing Avatar Viewport (PyQt6).

Integrates the high-fidelity Three.js PBR Mannequin Engine into the desktop application:
  - Procedural mannequin anatomy with PBR skin and uniform shaders
  - Cinematic studio key and rim lighting
  - 5-finger articulated hand & arm kinematics
  - Interactive JavaScript bridge for dynamic sign pose switching
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEB_ENGINE_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning("QWebEngineView could not be loaded: %s. Using fallback view.", e)
    WEB_ENGINE_AVAILABLE = False


class MannequinActionViewer(QWidget):
    """3D Realistic Mannequin Signing Avatar Viewport.

    Loads the high-fidelity Three.js PBR Mannequin Engine in an embedded WebEngine view.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.web_view: Optional[Any] = None

        # Sign mapping dictionary (Bengali & English glosses)
        self.sign_map = {
            "ধন্যবাদ": "thank_you",
            "thank_you": "thank_you",
            "thank you": "thank_you",
            "dhonnobad": "thank_you",
            "মা": "mother",
            "mother": "mother",
            "ma": "mother",
            "বাবা": "father",
            "father": "father",
            "baba": "father",
            "ডাক্তার": "doctor",
            "doctor": "doctor",
            "daktar": "doctor",
            "সালাম": "salam",
            "salam": "salam",
            "সাহায্য": "police",
            "help": "police",
            "sahajjo": "police",
            "পুলিশ": "police",
            "police": "police"
        }

        if WEB_ENGINE_AVAILABLE:
            try:
                self.web_view = QWebEngineView(self)
                layout.addWidget(self.web_view)

                # Locate the HTML avatar template in backend/templates/
                base_dir = Path(__file__).resolve().parents[3]
                html_path = base_dir / "backend" / "templates" / "mannequin_avatar.html"

                if html_path.exists():
                    self.web_view.load(QUrl.fromLocalFile(str(html_path)))
                else:
                    self.web_view.load(QUrl("http://127.0.0.1:8000/mannequin"))
            except Exception as ex:
                logger.error("Failed to initialize QWebEngineView: %s", ex)
                self._create_fallback_ui(layout)
        else:
            self._create_fallback_ui(layout)

    def _create_fallback_ui(self, layout: QVBoxLayout) -> None:
        """Fallback UI when WebEngine is not available or headless."""
        lbl = QLabel("৩ডি জীবন্ত অ্যাকশন অবতার (3D Mannequin Engine)", self)
        lbl.setStyleSheet("""
            background: #0A1122;
            color: #38BDF8;
            font-size: 13px;
            font-weight: bold;
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 8px;
            padding: 16px;
            qproperty-alignment: AlignCenter;
        """)
        layout.addWidget(lbl)

    def set_sign(self, gloss: str) -> str:
        """Switches the active animated sign on the 3D mannequin."""
        target_sign = self.sign_map.get(str(gloss).strip().lower(), self.sign_map.get(str(gloss).strip(), "police"))
        
        if self.web_view is not None:
            try:
                js_code = (
                    f"if (document.getElementById('sign-selector')) {{"
                    f"  document.getElementById('sign-selector').value = '{target_sign}';"
                    f"  document.getElementById('sign-selector').dispatchEvent(new Event('change'));"
                    f"}}"
                )
                self.web_view.page().runJavaScript(js_code)
            except Exception as e:
                logger.debug("Failed to run JavaScript on web_view: %s", e)

        return target_sign
