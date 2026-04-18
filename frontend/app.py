"""BitBot Streamlit UI — session chat with LangGraph + Bento classification."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx
import streamlit as st

BACKEND_DEFAULT = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str) -> datetime:
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_relative(iso_ts: str | None) -> str:
    """Human-readable relative time (e.g. just now, 45s ago)."""
    if not iso_ts:
        return ""
    try:
        dt = _parse_iso(iso_ts)
    except (ValueError, TypeError):
        return ""
    now = datetime.now(timezone.utc)
    secs = int((now - dt).total_seconds())
    if secs < 0:
        return "just now"
    if secs < 10:
        return "just now"
    if secs < 60:
        return f"{secs}s ago"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 7:
        return f"{days}d ago"
    return dt.strftime("%Y-%m-%d %H:%M")


def _msg_from_api(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": m.get("role", "user"),
        "content": m.get("content", ""),
        "metadata": m.get("metadata") if isinstance(m.get("metadata"), dict) else {},
        "created_at": m.get("created_at"),
    }


def _post_classify(
    text: str,
    *,
    session_id: str | None,
    full_flow: bool,
) -> dict:
    body: dict = {"text": text, "full_flow": full_flow}
    if session_id:
        body["session_id"] = session_id
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{BACKEND_DEFAULT}/classify", json=body)
        r.raise_for_status()
        return r.json()


def _post_escalation_decision(*, session_id: str, action_id: str, decision: str) -> dict:
    body = {"session_id": session_id, "action_id": action_id, "decision": decision}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{BACKEND_DEFAULT}/escalations/decision", json=body)
        r.raise_for_status()
        return r.json()


def main() -> None:
    st.set_page_config(page_title="BitBot — Support chat", layout="centered")
    st.title("BitBot — Support chat")
    st.caption(f"Backend: `{BACKEND_DEFAULT}`")

    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_classify_text" not in st.session_state:
        st.session_state.pending_classify_text = None
    if "last_classify_json" not in st.session_state:
        st.session_state.last_classify_json = None

    with st.sidebar:
        st.subheader("Session")
        use_full = st.toggle("Full flow (Postgres + LangGraph + LLM)", value=True)
        if st.session_state.last_classify_json and use_full:
            si = st.session_state.last_classify_json.get("session_issue") or {}
            if isinstance(si, dict):
                st.markdown("**Active issue**")
                st.caption("Intent (locked until resolved)")
                st.code(si.get("intent") or "—")
                st.caption("User request")
                st.write(si.get("user_request") or "—")
                st.caption("Resolved")
                st.write("Yes" if si.get("is_resolved") else "No")
        if st.button("New session"):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.pending_classify_text = None
            st.session_state.last_classify_json = None
            st.rerun()
        if st.session_state.session_id:
            st.code(st.session_state.session_id)

    prompt = st.chat_input("Type a message…")
    if prompt and prompt.strip() and not st.session_state.pending_classify_text:
        user_text = prompt.strip()
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_text,
                "metadata": {},
                "created_at": _now_iso(),
            }
        )
        st.session_state.pending_classify_text = user_text

    for m in st.session_state.messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        ts = m.get("created_at")
        rel = _format_relative(ts if isinstance(ts, str) else None)
        with st.chat_message(role):
            if rel:
                st.caption(rel)
            st.write(content)

    if st.session_state.pending_classify_text:
        user_text = st.session_state.pending_classify_text
        with st.chat_message("assistant"):
            with st.spinner("AI is processing…"):
                try:
                    out = _post_classify(
                        user_text,
                        session_id=st.session_state.session_id,
                        full_flow=use_full,
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Request failed: {e}")
                    st.session_state.pending_classify_text = None
                else:
                    st.session_state.session_id = out.get("session_id") or st.session_state.session_id

                    if use_full and out.get("messages"):
                        st.session_state.messages = [_msg_from_api(m) for m in out["messages"]]
                    else:
                        base = st.session_state.messages[:-1]
                        base.append(
                            {
                                "role": "user",
                                "content": user_text,
                                "metadata": {},
                                "created_at": _now_iso(),
                            }
                        )
                        if out.get("assistant_reply"):
                            base.append(
                                {
                                    "role": "assistant",
                                    "content": str(out["assistant_reply"]),
                                    "metadata": {},
                                    "created_at": _now_iso(),
                                }
                            )
                        else:
                            base.append(
                                {
                                    "role": "assistant",
                                    "content": (
                                        f"Category: `{out.get('category')}` · "
                                        f"confidence: `{out.get('confidence')}`"
                                    ),
                                    "metadata": {},
                                    "created_at": _now_iso(),
                                }
                            )
                        st.session_state.messages = base

                    st.session_state.last_classify_json = out
                    st.session_state.pending_classify_text = None
                    st.rerun()

    pending_action: dict | None = None
    for m in reversed(st.session_state.messages):
        md = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
        if md.get("pending_human_action"):
            pending_action = md
            break

    if pending_action and st.session_state.session_id:
        action_id = str(pending_action.get("action_id") or "")
        if action_id:
            col1, col2 = st.columns(2)
            if col1.button("Accept escalation", use_container_width=True):
                try:
                    out = _post_escalation_decision(
                        session_id=st.session_state.session_id,
                        action_id=action_id,
                        decision="accept",
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": str(out.get("assistant_reply") or ""),
                            "metadata": {
                                "action_id": action_id,
                                "decision": "accept",
                                "pending_human_action": False,
                            },
                            "created_at": out.get("created_at"),
                        }
                    )
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Escalation request failed: {e}")
            if col2.button("Reject escalation", use_container_width=True):
                try:
                    out = _post_escalation_decision(
                        session_id=st.session_state.session_id,
                        action_id=action_id,
                        decision="reject",
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": str(out.get("assistant_reply") or ""),
                            "metadata": {
                                "action_id": action_id,
                                "decision": "reject",
                                "pending_human_action": False,
                            },
                            "created_at": out.get("created_at"),
                        }
                    )
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Escalation request failed: {e}")

    if st.session_state.last_classify_json is not None:
        with st.expander("Last response (JSON)"):
            st.json(st.session_state.last_classify_json)


if __name__ == "__main__":
    main()
