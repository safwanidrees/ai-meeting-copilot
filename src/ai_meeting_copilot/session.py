"""Session history and export (Markdown for reading, JSON for tooling)."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Exchange:
    heard: str
    answer: str = ""
    follow_up: str = ""
    sources: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # seconds
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def reply(self) -> str:
        """The answer as the model originally formatted it (used as conversation memory)."""
        reply = f"ANSWER: {self.answer}"
        return f"{reply}\n\nFOLLOW-UP: {self.follow_up}" if self.follow_up else reply


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started_at: datetime = field(default_factory=datetime.now)
    exchanges: list[Exchange] = field(default_factory=list)

    def add(self, exchange: Exchange) -> None:
        self.exchanges.append(exchange)

    def history(self, turns: int) -> list[tuple[str, str]]:
        """Last `turns` answered exchanges as (heard, reply) pairs."""
        answered = [e for e in self.exchanges if e.answer]
        return [(e.heard, e.reply) for e in answered[-turns:]] if turns > 0 else []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "exchanges": [asdict(e) for e in self.exchanges],
        }

    def to_markdown(self) -> str:
        lines = [f"# Meeting session {self.id}", "", f"Started: {self.started_at:%Y-%m-%d %H:%M}", ""]
        for n, e in enumerate(self.exchanges, start=1):
            lines += [f"## {n}. {e.timestamp[11:]}", "", f"**They said:** {e.heard}", ""]
            if e.answer:
                lines += [f"**Answer:** {e.answer}", ""]
            if e.follow_up:
                lines += [f"**Likely follow-up:** {e.follow_up}", ""]
            if e.sources:
                lines += [f"_Sources: {', '.join(e.sources)}_", ""]
        return "\n".join(lines)

    def export(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"{self.started_at:%Y%m%d-%H%M%S}_{self.id}"
        md, js = stem.with_suffix(".md"), stem.with_suffix(".json")
        md.write_text(self.to_markdown(), encoding="utf-8")
        js.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return md, js
