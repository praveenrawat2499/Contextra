import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

CHAT_HISTORY_DIR = Path("data/chat_sessions")


def get_chat_history_dir() -> Path:
    """Ensure and return the directory path for saved chat sessions."""
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return CHAT_HISTORY_DIR


def create_new_session_id() -> str:
    """Generate a unique session identifier."""
    return f"session_{uuid.uuid4().hex[:8]}"


def get_session_file_path(session_id: str) -> Path:
    """Get the JSON file path for a session ID."""
    clean_id = os.path.basename(session_id)
    return get_chat_history_dir() / f"{clean_id}.json"


def save_session(
    session_id: str,
    title: str,
    messages: List[Dict[str, Any]],
    created_at: Optional[str] = None,
) -> None:
    """Save a chat session and its messages to a JSON file."""
    get_chat_history_dir()
    filepath = get_session_file_path(session_id)

    if not created_at and filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
                created_at = existing.get("created_at")
        except Exception:
            created_at = None

    if not created_at:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    payload = {
        "session_id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a single chat session by session_id."""
    filepath = get_session_file_path(session_id)
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_all_sessions() -> Dict[str, Dict[str, Any]]:
    """Load all saved sessions from disk, sorted by most recently updated."""
    directory = get_chat_history_dir()
    sessions = {}

    for file_path in directory.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sid = data.get("session_id", file_path.stem)
                sessions[sid] = data
        except Exception:
            continue

    # Sort by updated_at descending
    sorted_sessions = dict(
        sorted(
            sessions.items(),
            key=lambda item: item[1].get("updated_at", ""),
            reverse=True,
        )
    )
    return sorted_sessions


def delete_session(session_id: str) -> bool:
    """Delete a session JSON file from disk."""
    filepath = get_session_file_path(session_id)
    if filepath.exists():
        try:
            filepath.unlink()
            return True
        except Exception:
            return False
    return False


def clear_all_sessions() -> int:
    """Delete all session JSON files from disk. Returns count of deleted files."""
    directory = get_chat_history_dir()
    deleted = 0
    for file_path in directory.glob("*.json"):
        try:
            file_path.unlink()
            deleted += 1
        except Exception:
            continue
    return deleted
