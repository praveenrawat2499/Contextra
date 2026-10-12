import json
from pathlib import Path

from src import chat_history


def test_session_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_history, "CHAT_HISTORY_DIR", tmp_path / "chat_sessions")

    # 1. Create session id
    sid = chat_history.create_new_session_id()
    assert sid.startswith("session_")

    # 2. Save session
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {
            "role": "assistant",
            "content": "ML is a field of AI.",
            "citations": [{"citation_index": 1, "source": "doc1.pdf", "chunk_id": "abc", "snippet": "ML field"}],
        },
    ]
    chat_history.save_session(
        session_id=sid,
        title="What is machine learning?",
        messages=messages,
    )

    # 3. Load session
    loaded = chat_history.load_session(sid)
    assert loaded is not None
    assert loaded["session_id"] == sid
    assert loaded["title"] == "What is machine learning?"
    assert len(loaded["messages"]) == 2
    assert loaded["messages"][1]["citations"][0]["source"] == "doc1.pdf"

    # 4. Load all sessions
    all_sessions = chat_history.load_all_sessions()
    assert sid in all_sessions
    assert len(all_sessions) == 1

    # 5. Save another session
    sid2 = chat_history.create_new_session_id()
    chat_history.save_session(
        session_id=sid2,
        title="Second chat",
        messages=[{"role": "user", "content": "Hello"}],
    )
    all_sessions2 = chat_history.load_all_sessions()
    assert len(all_sessions2) == 2

    # 6. Delete single session
    deleted = chat_history.delete_session(sid)
    assert deleted is True
    assert chat_history.load_session(sid) is None
    assert len(chat_history.load_all_sessions()) == 1

    # 7. Clear all sessions
    count = chat_history.clear_all_sessions()
    assert count == 1
    assert len(chat_history.load_all_sessions()) == 0
