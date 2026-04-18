"""BitBot Streamlit UI — session chat with LangGraph + Bento classification."""

from __future__ import annotations

import os

import httpx
import streamlit as st

BACKEND_DEFAULT = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")


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

    with st.sidebar:
        st.subheader("Session")
        use_full = st.toggle("Full flow (Postgres + LangGraph + LLM)", value=True)
        if st.button("New session"):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()
        if st.session_state.session_id:
            st.code(st.session_state.session_id)

    for m in st.session_state.messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        with st.chat_message(role):
            st.write(content)

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
                        }
                    )
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Escalation request failed: {e}")

    prompt = st.chat_input("Type a message…")
    if not prompt or not prompt.strip():
        return

    user_text = prompt.strip()

    try:
        out = _post_classify(
            user_text,
            session_id=st.session_state.session_id,
            full_flow=use_full,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Request failed: {e}")
        return

    st.session_state.session_id = out.get("session_id") or st.session_state.session_id

    if use_full and out.get("messages"):
        st.session_state.messages = [
            {
                "role": m.get("role", "user"),
                "content": m.get("content", ""),
                "metadata": m.get("metadata") if isinstance(m.get("metadata"), dict) else {},
            }
            for m in out["messages"]
        ]
    else:
        st.session_state.messages.append({"role": "user", "content": user_text})
        if out.get("assistant_reply"):
            st.session_state.messages.append(
                {"role": "assistant", "content": str(out["assistant_reply"])}
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Category: `{out.get('category')}` · "
                        f"confidence: `{out.get('confidence')}`"
                    ),
                }
            )

    with st.expander("Last response (JSON)"):
        st.json(out)


if __name__ == "__main__":
    main()
