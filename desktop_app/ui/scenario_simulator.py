import logging
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QListWidget, QListWidgetItem, QSplitter, QTextEdit,
    QStackedWidget, QFrame
)

logger = logging.getLogger(__name__)

# Modern Palette
BG_COLOR = "#1E1E2E"
TEXT_COLOR = "#CDD6F4"
ACCENT_COLOR = "#F9E2AF"
SUCCESS_COLOR = "#A6E3A1"
ERROR_COLOR = "#F38BA8"
SURFACE_COLOR = "#313244"
PANEL_COLOR = "#181825"

SCENARIOS = {
    "Hospital": [
        {"actor": "AI", "text": "আপনি কেমন অনুভব করছেন? (How are you feeling?)"},
        {"actor": "User", "text": "আমার মাথা ব্যথা করছে। (I have a headache.)"},
        {"actor": "AI", "text": "কতদিন ধরে? (For how long?)"},
        {"actor": "User", "text": "দুই দিন ধরে। (For two days.)"}
    ],
    "Police 999": [
        {"actor": "AI", "text": "৯৯৯ জরুরি সেবা, কীভাবে সাহায্য করতে পারি? (999 emergency, how can I help?)"},
        {"actor": "User", "text": "আমি একটি দুর্ঘটনা দেখেছি। (I saw an accident.)"},
        {"actor": "AI", "text": "কোথায়? (Where?)"},
    ],
    "Banking": [
        {"actor": "User", "text": "আমি একটি অ্যাকাউন্ট খুলতে চাই। (I want to open an account.)"},
        {"actor": "AI", "text": "আপনার জাতীয় পরিচয়পত্র আছে? (Do you have your NID?)"},
    ],
    "Classroom": [
        {"actor": "User", "text": "আমি বুঝতে পারিনি। (I did not understand.)"},
        {"actor": "AI", "text": "কোন অংশটি? (Which part?)"},
    ]
}

class ScenarioSimulator(QWidget):
    """Public & Emergency Scenario Simulator."""
    
    request_back = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.current_scenario = None
        self.current_step = 0
        self._init_ui()
        
    def _init_ui(self):
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QListWidget {{
                background-color: {PANEL_COLOR};
                border: 1px solid {SURFACE_COLOR};
                border-radius: 8px;
                padding: 5px;
                font-size: 16px;
            }}
            QListWidget::item:selected {{
                background-color: {ACCENT_COLOR};
                color: #11111B;
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: {SURFACE_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {ACCENT_COLOR};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_COLOR};
                color: #11111B;
            }}
            QTextEdit {{
                background-color: {PANEL_COLOR};
                border: 1px solid {SURFACE_COLOR};
                border-radius: 8px;
                color: {TEXT_COLOR};
                font-size: 16px;
                padding: 10px;
            }}
        """)
        
        main_layout = QVBoxLayout(self)
        
        # --- Top Header ---
        header_layout = QHBoxLayout()
        back_btn = QPushButton("← Back to Main")
        back_btn.clicked.connect(self.request_back.emit)
        
        title_lbl = QLabel("Scenario & Emergency Simulator")
        title_lbl.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title_lbl.setStyleSheet(f"color: {ACCENT_COLOR};")
        
        header_layout.addWidget(back_btn)
        header_layout.addStretch()
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # --- Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel (Scenarios)
        scenario_panel = QWidget()
        scenario_layout = QVBoxLayout(scenario_panel)
        scenario_layout.setContentsMargins(0, 0, 0, 0)
        
        s_header = QLabel("Scenarios")
        s_header.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        scenario_layout.addWidget(s_header)
        
        self.scenario_list = QListWidget()
        for s in SCENARIOS.keys():
            self.scenario_list.addItem(s)
        self.scenario_list.itemClicked.connect(self._load_scenario)
        scenario_layout.addWidget(self.scenario_list)
        
        # Right Panel (Chat / Guided Practice)
        practice_panel = QWidget()
        practice_layout = QVBoxLayout(practice_panel)
        practice_layout.setContentsMargins(10, 0, 0, 0)
        
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        practice_layout.addWidget(self.chat_history)
        
        self.prompt_label = QLabel("Select a scenario to start...")
        self.prompt_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt_label.setStyleSheet(f"color: {SUCCESS_COLOR}; padding: 10px;")
        practice_layout.addWidget(self.prompt_label)
        
        btn_layout = QHBoxLayout()
        self.next_btn = QPushButton("Next Dialogue Step")
        self.next_btn.clicked.connect(self._advance_dialogue)
        self.next_btn.setEnabled(False)
        btn_layout.addWidget(self.next_btn)
        practice_layout.addLayout(btn_layout)
        
        splitter.addWidget(scenario_panel)
        splitter.addWidget(practice_panel)
        splitter.setSizes([300, 700])
        
        main_layout.addWidget(splitter)
        
    def _load_scenario(self, item: QListWidgetItem):
        self.current_scenario = item.text()
        self.current_step = 0
        self.chat_history.clear()
        self.chat_history.append(f"<h3 style='color:{ACCENT_COLOR};'>Loaded Scenario: {self.current_scenario}</h3>")
        self.next_btn.setEnabled(True)
        self._advance_dialogue()
        
    def _advance_dialogue(self):
        if not self.current_scenario:
            return
            
        dialogue = SCENARIOS[self.current_scenario]
        
        if self.current_step < len(dialogue):
            line = dialogue[self.current_step]
            actor = line["actor"]
            text = line["text"]
            
            if actor == "AI":
                self.chat_history.append(f"<b style='color:{ACCENT_COLOR};'>Agent:</b> {text}")
                self.prompt_label.setText("Your Turn to Sign...")
                self.prompt_label.setStyleSheet(f"color: {ERROR_COLOR}; padding: 10px;")
            else:
                self.chat_history.append(f"<b style='color:{SUCCESS_COLOR};'>You:</b> {text}")
                self.prompt_label.setText("Agent is responding...")
                self.prompt_label.setStyleSheet(f"color: {SUCCESS_COLOR}; padding: 10px;")
                
            self.current_step += 1
        else:
            self.prompt_label.setText("Scenario Completed!")
            self.prompt_label.setStyleSheet(f"color: {SUCCESS_COLOR}; padding: 10px;")
            self.next_btn.setEnabled(False)
