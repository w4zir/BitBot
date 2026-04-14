"""BitBot Streamlit UI — issue / no_issue classification demo."""

from __future__ import annotations

import os

import httpx
import streamlit as st

BACKEND_DEFAULT = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")


def _classify(text: str) -> dict:
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{BACKEND_DEFAULT}/classify", json={"text": text})
        r.raise_for_status()
        return r.json()


def main() -> None:
    st.set_page_config(page_title="BitBot — Issue classifier", layout="centered")
    st.title("BitBot — Issue / no_issue")
    st.caption(f"Backend: `{BACKEND_DEFAULT}`")

    text = st.text_area("Message", placeholder="Type a customer message…", height=120)
    if st.button("Classify", type="primary"):
        if not text.strip():
            st.warning("Enter some text.")
            return
        try:
            out = _classify(text)
            st.success("Result")
            st.json(out)
        except Exception as e:  # noqa: BLE001
            st.error(f"Request failed: {e}")


if __name__ == "__main__":
    main()
