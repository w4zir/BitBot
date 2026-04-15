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
            {"role": m.get("role", "user"), "content": m.get("content", "")}
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
