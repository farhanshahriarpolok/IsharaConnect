"""Design System Theme Definitions for IsharaConnect."""

class ThemeColors:
    # Deep Slate
    BG_DARK = "#0F172A"
    PANEL_DARK = "#1E293B"
    SURFACE_DARK = "#334155"
    
    # Text
    TEXT_PRIMARY = "#F8FAFC"
    TEXT_SECONDARY = "#94A3B8"
    
    # Accents
    CYAN_ACCENT = "#06B6D4"
    EMERALD_SUCCESS = "#10B981"
    CORAL_ERROR = "#F43F5E"
    
    # Glassmorphism utilities
    GLASS_BG = "rgba(30, 41, 59, 0.7)"
    GLASS_BORDER = "rgba(255, 255, 255, 0.1)"

class ThemeStyles:
    @staticmethod
    def get_global_stylesheet() -> str:
        return f"""
        QWidget {{
            background-color: {ThemeColors.BG_DARK};
            color: {ThemeColors.TEXT_PRIMARY};
            font-family: 'Segoe UI', 'Inter', 'SolaimanLipi', sans-serif;
        }}
        
        QFrame, QWidget#GlassCard {{
            background-color: {ThemeColors.PANEL_DARK};
            border: 1px solid {ThemeColors.GLASS_BORDER};
            border-radius: 12px;
        }}
        
        QPushButton {{
            background-color: {ThemeColors.CYAN_ACCENT};
            color: {ThemeColors.BG_DARK};
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 16px;
            border: none;
        }}
        
        QPushButton:hover {{
            background-color: #22D3EE;
        }}
        
        QPushButton:disabled {{
            background-color: {ThemeColors.SURFACE_DARK};
            color: {ThemeColors.TEXT_SECONDARY};
        }}
        
        QLineEdit, QTextEdit, QComboBox {{
            background-color: {ThemeColors.BG_DARK};
            border: 1px solid {ThemeColors.SURFACE_DARK};
            border-radius: 6px;
            padding: 8px;
            color: {ThemeColors.TEXT_PRIMARY};
        }}
        
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
            border: 1px solid {ThemeColors.CYAN_ACCENT};
        }}
        """
