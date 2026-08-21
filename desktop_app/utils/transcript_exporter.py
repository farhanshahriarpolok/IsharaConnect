"""Conversation Transcript and Chat History Exporter for IsharaConnect.

Supports exporting conversation transcripts to:
1. Formatted Plain Text (.txt)
2. Structured JSON (.json)
3. Styled PDF Document (.pdf) using ReportLab
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)


def _sanitize_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures message has required fields: timestamp, sender, and text."""
    timestamp = msg.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sender = msg.get("sender") or msg.get("role") or "Participant"
    text = msg.get("text") or msg.get("content") or msg.get("message") or ""
    
    # Strip HTML tags if present (e.g., <b>Speaker:</b>)
    import re
    clean_text = re.sub(r"<[^>]+>", "", str(text)).strip()

    return {
        "timestamp": str(timestamp),
        "sender": str(sender),
        "text": clean_text,
        "raw_text": str(text),
    }


def export_to_txt(
    messages: List[Dict[str, Any]],
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Exports messages to a formatted text file.

    Args:
        messages: List of message dictionaries.
        file_path: Target destination path.
        metadata: Optional dictionary of session info.

    Returns:
        The target file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = metadata or {}
    export_time = meta.get("export_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    room_id = meta.get("room_id", "Default Room")
    session_mode = meta.get("mode", "Two-Way Duplex")

    lines = [
        "=" * 70,
        "ISHARACONNECT - TWO-WAY COMMUNICATION CHAT TRANSCRIPT",
        "=" * 70,
        f"Export Date: {export_time}",
        f"Room ID:     {room_id}",
        f"Session Mode:{session_mode}",
        f"Total Logged Messages: {len(messages)}",
        "-" * 70,
        "",
    ]

    for msg in messages:
        sanitized = _sanitize_message(msg)
        ts = sanitized["timestamp"]
        sender = sanitized["sender"].upper()
        text = sanitized["text"]
        lines.append(f"[{ts}] [{sender}]: {text}")

    lines.extend(["", "=" * 70, "END OF TRANSCRIPT", "=" * 70])

    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Exported %d messages to TXT: %s", len(messages), path)
    return str(path.resolve())


def export_to_json(
    messages: List[Dict[str, Any]],
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Exports messages to a structured JSON file.

    Args:
        messages: List of message dictionaries.
        file_path: Target destination path.
        metadata: Optional dictionary of session info.

    Returns:
        The target file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = metadata or {}
    export_time = meta.get("export_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    room_id = meta.get("room_id", "Default Room")
    session_mode = meta.get("mode", "Two-Way Duplex")

    sanitized_messages = [_sanitize_message(m) for m in messages]

    payload = {
        "isharaconnect_version": "2.0.0",
        "export_timestamp": export_time,
        "session": {
            "room_id": room_id,
            "mode": session_mode,
            "total_messages": len(sanitized_messages),
        },
        "metadata": meta,
        "messages": sanitized_messages,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Exported %d messages to JSON: %s", len(messages), path)
    return str(path.resolve())


def export_to_pdf(
    messages: List[Dict[str, Any]],
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    title: str = "IsharaConnect Chat Transcript",
) -> str:
    """Exports messages to a professionally styled PDF document using ReportLab.

    Args:
        messages: List of message dictionaries.
        file_path: Target destination path.
        metadata: Optional dictionary of session info.
        title: Title of the document.

    Returns:
        The target file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "TranscriptTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=HexColor("#0F172A"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "TranscriptSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=14,
        textColor=HexColor("#0D9488"),
        spaceAfter=10,
    )

    meta_style = ParagraphStyle(
        "TranscriptMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=HexColor("#475569"),
    )

    sender_signer_style = ParagraphStyle(
        "SenderSigner",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=HexColor("#059669"),
    )

    sender_speaker_style = ParagraphStyle(
        "SenderSpeaker",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=HexColor("#2563EB"),
    )

    sender_default_style = ParagraphStyle(
        "SenderDefault",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=HexColor("#64748B"),
    )

    text_style = ParagraphStyle(
        "MessageText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=HexColor("#1E293B"),
    )

    time_style = ParagraphStyle(
        "MessageTime",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=HexColor("#94A3B8"),
    )

    story = []

    # Title & Subtitle Header
    story.append(Paragraph(title, title_style))
    story.append(
        Paragraph("Real-Time Bangla Sign Language (BdSL) & Vocalized Speech Session Record", subtitle_style)
    )
    story.append(HRFlowable(width="100%", thickness=2, color=HexColor("#06B6D4"), spaceAfter=10))

    # Metadata Block
    meta = metadata or {}
    export_time = meta.get("export_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    room_id = meta.get("room_id", "Default Public Room")
    session_mode = meta.get("mode", "Duplex (Signer ↔ Speaker)")

    meta_text = (
        f"<b>Session Room:</b> {room_id} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Mode:</b> {session_mode} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Export Timestamp:</b> {export_time} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Message Count:</b> {len(messages)}"
    )
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 12))

    # Messages Table
    table_data = [
        [
            Paragraph("<b>Timestamp</b>", meta_style),
            Paragraph("<b>Participant</b>", meta_style),
            Paragraph("<b>Message Content</b>", meta_style),
        ]
    ]

    if not messages:
        table_data.append([
            Paragraph("-", time_style),
            Paragraph("System", sender_default_style),
            Paragraph("<i>No messages recorded in this session.</i>", text_style),
        ])
    else:
        for msg in messages:
            sanitized = _sanitize_message(msg)
            sender_lower = sanitized["sender"].lower()
            if "signer" in sender_lower or "sign" in sender_lower:
                s_style = sender_signer_style
                sender_label = f"🖐️ {sanitized['sender']}"
            elif "speaker" in sender_lower or "voice" in sender_lower:
                s_style = sender_speaker_style
                sender_label = f"🗣️ {sanitized['sender']}"
            elif "you" in sender_lower:
                s_style = sender_signer_style
                sender_label = f"👤 {sanitized['sender']}"
            else:
                s_style = sender_default_style
                sender_label = sanitized["sender"]

            # Format text safely for ReportLab Paragraph (escape special XML chars if needed)
            safe_text = (
                sanitized["text"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            # Split short time if full date present
            ts_str = sanitized["timestamp"]
            if " " in ts_str:
                ts_display = ts_str.split(" ")[1]
            else:
                ts_display = ts_str

            table_data.append([
                Paragraph(ts_display, time_style),
                Paragraph(sender_label, s_style),
                Paragraph(safe_text, text_style),
            ])

    col_widths = [1.1 * inch, 1.6 * inch, 4.8 * inch]
    msg_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    msg_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#F1F5F9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#0F172A")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), 1.5, HexColor("#CBD5E1")),
            ("LINEBELOW", (0, 1), (-1, -1), 0.5, HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F8FAFC")]),
        ])
    )
    story.append(msg_table)
    story.append(Spacer(1, 15))

    # Footer notice
    footer_text = "IsharaConnect - Empowering Seamless Communication for Bangladesh's Deaf and Mute Community."
    story.append(Paragraph(footer_text, subtitle_style))

    doc.build(story)

    logger.info("Exported %d messages to PDF: %s", len(messages), path)
    return str(path.resolve())
