"""Foreman review console.

The screen a safety supervisor would actually stand in front of. Three questions, in
the order they get asked: what needs my attention, why does the system think so, and
did anything happen that the taxonomy does not cover.

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# On Streamlit Community Cloud the key arrives via st.secrets rather than the
# environment. Bridging it here keeps foreman.nim free of any Streamlit import, so the
# pipeline, the MCP server and the eval harness stay runnable without it.
if "NVIDIA_API_KEY" not in os.environ:
    try:
        if "NVIDIA_API_KEY" in st.secrets:
            os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]
    except Exception:  # noqa: BLE001 - no secrets file locally is normal
        pass

from foreman import verify  # noqa: E402
from foreman.hazards import BY_KEY  # noqa: E402
from foreman.index import TimelineIndex  # noqa: E402
from foreman.perception import load as load_perceptions  # noqa: E402

RUNS = ROOT / "data" / "runs"
SEV_COLOR = {"high": "#e5484d", "medium": "#f5a623", "low": "#8b8d98"}

st.set_page_config(page_title="Foreman", page_icon="🦺", layout="wide")


def ts(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


@st.cache_data(show_spinner=False)
def load_run(video_id: str):
    run = RUNS / video_id
    return (
        json.loads((run / "summary.json").read_text()),
        [a.as_dict() for a in verify.load(run / "alerts.json")],
        [p.as_dict() for p in load_perceptions(run / "perceptions.json")],
    )


def available_runs() -> list[str]:
    """Processed runs, busiest queue first.

    A reviewer opening this console wants the shift that needs work, not the one that
    sorts first alphabetically.
    """
    if not RUNS.exists():
        return []
    names = [p.name for p in RUNS.iterdir() if (p / "summary.json").exists()]
    def confirmed(name: str) -> int:
        return json.loads((RUNS / name / "summary.json").read_text()).get("confirmed", 0)
    return sorted(names, key=lambda n: (-confirmed(n), n))


runs = available_runs()
if not runs:
    st.title("Foreman")
    st.warning("No processed videos yet. Run `python -m foreman.pipeline <video.mp4>` first.")
    st.stop()

st.markdown(
    "<h1 style='margin-bottom:0'>Foreman</h1>"
    "<p style='color:#8b8d98;margin-top:4px'>Agentic vision review for warehouse "
    "safety footage, on NVIDIA NIM</p>",
    unsafe_allow_html=True,
)

video_id = st.sidebar.selectbox("Processed video", runs)
summary, alerts, perceptions = load_run(video_id)

confirmed = [a for a in alerts if a["verdict"] == "confirmed"]
rejected = [a for a in alerts if a["verdict"] == "rejected"]

st.sidebar.markdown("### Models")
for k, v in summary.get("models", {}).items():
    st.sidebar.caption(f"**{k}** · `{v}`")
st.sidebar.markdown("### This run")
st.sidebar.caption(
    f"{summary['windows']} windows · {summary['wall_seconds']:.0f}s wall clock\n\n"
    f"{summary['usage']['calls']} NIM calls · "
    f"{summary['usage']['total_tokens']:,} tokens"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Windows analysed", summary["windows"])
c2.metric("Candidates raised", summary["candidates"])
c3.metric("Confirmed alerts", len(confirmed))
c4.metric(
    "Filtered out",
    f"{len(rejected) + summary['gated_non_footage']}",
    help="Detections the verifier rejected, plus windows gated as non-footage "
         "(title cards, slides, presenters).",
)

tab_queue, tab_search, tab_audit = st.tabs(
    ["Alert queue", "Search the footage", "Audit what was filtered"]
)

with tab_queue:
    if not confirmed:
        st.success("No hazards confirmed in this footage.")
    order = {"high": 0, "medium": 1, "low": 2}
    for a in sorted(confirmed, key=lambda x: (order.get(x["severity"], 1), x["start_s"])):
        color = SEV_COLOR.get(a["severity"], "#8b8d98")
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:2px 0 2px 12px;margin-top:14px'>"
            f"<span style='color:{color};font-weight:700;text-transform:uppercase;"
            f"font-size:12px;letter-spacing:.06em'>{a['severity']}</span>"
            f"<span style='color:#8b8d98;font-size:12px'> &nbsp;·&nbsp; {ts(a['start_s'])}"
            f"–{ts(a['end_s'])}</span><br>"
            f"<span style='font-size:18px;font-weight:600'>{a['label']}</span></div>",
            unsafe_allow_html=True,
        )
        left, right = st.columns([2, 3])
        clip = RUNS / video_id / "clips" / f"{a['alert_id'].replace(':', '_')}.mp4"
        with left:
            if clip.exists():
                st.video(str(clip))
            else:
                # Expected on the hosted demo. The clips are excerpts of third-party
                # training footage, so they are cut locally and never committed; see
                # DATA.md. Saying so beats an empty box that reads as a bug.
                st.info(
                    "Evidence clip available when run locally.\n\n"
                    "Clips are cut from third-party footage that this repo does not "
                    "redistribute. `python scripts/fetch_data.py` then re-run the "
                    "pipeline to see them."
                )
        with right:
            st.markdown(f"**Why this was confirmed** — {a['rationale']}")
            with st.expander("Evidence chain"):
                st.markdown(f"**Proposing model claimed:** {a['description']}")
                st.markdown(f"**Cited evidence:** {a['evidence']}")
                st.markdown(f"**Proposing confidence:** {a['vlm_confidence']:.2f}")
                hz = BY_KEY.get(a["type"])
                if hz:
                    st.markdown(f"**Bar for this class:** {hz.requires}")
                st.markdown(f"**Standard:** `{a['standard']}`")
                st.markdown(f"**Verified by:** `{a['verifier_model']}`")

with tab_search:
    st.caption(
        "Searches every analysed window, including ones that raised no alert. This is "
        "how you ask questions the hazard taxonomy was never designed to answer."
    )
    q = st.text_input(
        "What are you looking for?",
        placeholder="someone walking near a moving forklift",
    )
    if q:
        try:
            with st.spinner("searching"):
                idx = TimelineIndex.load(RUNS / video_id / "index")
                hits = idx.search(q, k=8)
        except Exception as exc:  # noqa: BLE001
            # Search is the one tab that calls out at request time, so it is also the
            # one that can fail in front of a visitor. Name the cause rather than
            # dumping a stack trace.
            st.error(
                f"Search needs a live NVIDIA NIM key to embed the query. {exc}"
            )
            hits = []
        for h in hits:
            st.markdown(
                f"**{ts(h.start_s)}–{ts(h.end_s)}** &nbsp; "
                f"<span style='color:#8b8d98'>relevance {h.score:.2f}</span>",
                unsafe_allow_html=True,
            )
            st.write(h.caption)
            if h.hazards:
                st.caption("candidates raised here: " + ", ".join(h.hazards))
            st.divider()

with tab_audit:
    st.caption(
        "Every detection the system removed, and why. A safety tool that cannot show "
        "what it suppressed is a safety tool nobody should trust."
    )
    gated = [p for p in perceptions if not p["is_camera_footage"]]
    if gated:
        st.markdown(f"#### {len(gated)} windows gated as non-footage")
        for p in gated:
            st.markdown(
                f"**{ts(p['start_s'])}–{ts(p['end_s'])}** · `{p['frame_content']}` — "
                f"{p['scene']}"
            )
        st.divider()
    st.markdown(f"#### {len(rejected)} detections rejected by the verifier")
    for a in rejected:
        st.markdown(
            f"**{ts(a['start_s'])}** · {a['label']} "
            f"<span style='color:#8b8d98'>(proposed at "
            f"{a['vlm_confidence']:.2f} confidence)</span>",
            unsafe_allow_html=True,
        )
        st.write(a["rationale"])
        if a["missing_evidence"]:
            st.caption(f"looked for and did not find: {a['missing_evidence']}")
        st.divider()
