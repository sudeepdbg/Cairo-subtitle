"""
core/subtitle_parser.py
Parses SRT, VTT subtitle files and YouTube transcripts into
a unified list of CueEntry objects with ms-precision timing.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CueEntry:
    index: int
    start_ms: int
    end_ms: int
    text: str
    raw: str = field(default="", repr=False)

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_sec(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    def fmt_time(self, ms: int) -> str:
        h = ms // 3_600_000
        m = (ms % 3_600_000) // 60_000
        s = (ms % 60_000) // 1000
        ms_r = ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d}.{ms_r:03d}"

    @property
    def start_fmt(self) -> str:
        return self.fmt_time(self.start_ms)

    @property
    def end_fmt(self) -> str:
        return self.fmt_time(self.end_ms)


def _ts_to_ms(ts: str) -> int:
    """Convert HH:MM:SS,mmm or HH:MM:SS.mmm or MM:SS.mmm to milliseconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, rest = parts
        s, ms = (rest.split(".") + ["0"])[:2]
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms.ljust(3, "0")[:3])
    elif len(parts) == 2:
        m, rest = parts
        s, ms = (rest.split(".") + ["0"])[:2]
        return (int(m) * 60 + int(s)) * 1000 + int(ms.ljust(3, "0")[:3])
    return 0


def _clean_text(t: str) -> str:
    """Remove HTML tags, WebVTT tags, and position cues."""
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\{[^}]+\}", "", t)
    t = re.sub(r"NOTE\s.*", "", t, flags=re.MULTILINE)
    t = re.sub(r"align:\w+\s+line:\S+\s+position:\S+\s+size:\S+", "", t)
    return " ".join(t.split())


def parse_srt(content: str) -> list[CueEntry]:
    """Parse SRT subtitle file content."""
    cues: list[CueEntry] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1],
        )
        if not time_match:
            continue
        start_ms = _ts_to_ms(time_match.group(1))
        end_ms = _ts_to_ms(time_match.group(2))
        text = _clean_text(" ".join(lines[2:]))
        if text:
            cues.append(CueEntry(idx, start_ms, end_ms, text, raw=block))
    return cues


def parse_vtt(content: str) -> list[CueEntry]:
    """Parse WebVTT subtitle file content."""
    # Strip WEBVTT header
    content = re.sub(r"^WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
    cues: list[CueEntry] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    idx = 1
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        # Optional cue identifier line
        start_line = 0
        if "-->" not in lines[0] and len(lines) > 1:
            start_line = 1
        if start_line >= len(lines):
            continue
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})",
            lines[start_line],
        )
        if not time_match:
            continue
        start_ms = _ts_to_ms(time_match.group(1))
        end_ms = _ts_to_ms(time_match.group(2))
        text = _clean_text(" ".join(lines[start_line + 1:]))
        if text:
            cues.append(CueEntry(idx, start_ms, end_ms, text, raw=block))
            idx += 1
    return cues


def parse_youtube_transcript(transcript_data: list[dict]) -> list[CueEntry]:
    """
    Convert YouTube Transcript API output to CueEntry list.
    Each entry is: {'text': str, 'start': float, 'duration': float}
    """
    cues = []
    for i, entry in enumerate(transcript_data):
        start_ms = int(entry["start"] * 1000)
        dur_ms = int(entry.get("duration", 2.0) * 1000)
        end_ms = start_ms + dur_ms
        text = _clean_text(entry.get("text", ""))
        if text:
            cues.append(CueEntry(i + 1, start_ms, end_ms, text))
    return cues


def parse_subtitle_file(content: str, fmt: Optional[str] = None) -> list[CueEntry]:
    """
    Auto-detect format and parse. fmt can be 'srt', 'vtt', or None (auto-detect).
    """
    content = content.strip()
    if fmt == "vtt" or content.startswith("WEBVTT"):
        return parse_vtt(content)
    # Auto-detect SRT vs VTT
    if re.match(r"^\d+\s*\n\d{2}:\d{2}:\d{2}", content):
        return parse_srt(content)
    if "WEBVTT" in content[:100] or re.match(r"^\d{2}:\d{2}", content):
        return parse_vtt(content)
    # Fallback: try SRT
    result = parse_srt(content)
    return result if result else parse_vtt(content)
