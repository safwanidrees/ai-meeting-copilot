import json

from ai_meeting_copilot.session import Exchange, Session


def test_history_skips_unanswered_and_limits_turns():
    s = Session()
    s.add(Exchange("caption only"))
    for i in range(5):
        s.add(Exchange(f"q{i}", answer=f"a{i}", follow_up=f"f{i}"))
    hist = s.history(2)
    assert hist == [("q3", "ANSWER: a3\n\nFOLLOW-UP: f3"), ("q4", "ANSWER: a4\n\nFOLLOW-UP: f4")]
    assert s.history(0) == []


def test_export_writes_markdown_and_json(tmp_path):
    s = Session()
    s.add(Exchange("What is RAG?", answer="Retrieval-augmented generation.", follow_up="Why hybrid?", sources=["notes.md"]))
    md, js = s.export(tmp_path)
    assert "**They said:** What is RAG?" in md.read_text()
    data = json.loads(js.read_text())
    assert data["id"] == s.id and data["exchanges"][0]["sources"] == ["notes.md"]
