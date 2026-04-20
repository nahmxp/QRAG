"""
QRAG Streamlit App – two-tab layout
  Tab 1 — Deep Reasoning Chat (RAG)
  Tab 2 — Quran Reader (PDF viewer with surah navigation)
"""

import streamlit as st
import tempfile, os, base64
from pathlib import Path

import pymupdf
from rag_engine import KnowledgeBase, ReasoningRAG

# ── Config ─────────────────────────────────────────────────────────────────────
QURAN_PDF = Path("PDF_Data/quran-english-translation-clearquran-edition-allah.pdf")

# All 114 surahs: (number, English name, Arabic name, start page in this PDF)
SURAHS = [
    (1,  "The Opening",        "Al-Fatihah",    12),
    (2,  "The Heifer",         "Al-Baqarah",    13),
    (3,  "Family of Imran",    "Ali Imran",     28),
    (4,  "Women",              "An-Nisa",       38),
    (5,  "The Table",          "Al-Ma'idah",    48),
    (6,  "Livestock",          "Al-An'am",      55),
    (7,  "The Elevations",     "Al-A'raf",      64),
    (8,  "The Spoils",         "Al-Anfal",      74),
    (9,  "Repentance",         "At-Tawbah",     77),
    (10, "Jonah",              "Yunus",         84),
    (11, "Hud",                "Hud",           89),
    (12, "Joseph",             "Yusuf",         95),
    (13, "Thunder",            "Ar-Ra'd",       100),
    (14, "Abraham",            "Ibrahim",       102),
    (15, "The Rock",           "Al-Hijr",       105),
    (16, "The Bee",            "An-Nahl",       107),
    (17, "The Night Journey",  "Al-Isra",       113),
    (18, "The Cave",           "Al-Kahf",       117),
    (19, "Mary",               "Maryam",        122),
    (20, "Ta-Ha",              "Ta-Ha",         125),
    (21, "The Prophets",       "Al-Anbiya",     130),
    (22, "The Pilgrimage",     "Al-Hajj",       134),
    (23, "The Believers",      "Al-Mu'minun",   137),
    (24, "The Light",          "An-Nur",        141),
    (25, "The Criterion",      "Al-Furqan",     144),
    (26, "The Poets",          "Ash-Shu'ara",   147),
    (27, "The Ant",            "An-Naml",       152),
    (28, "History",            "Al-Qasas",      156),
    (29, "The Spider",         "Al-Ankabut",    160),
    (30, "The Romans",         "Ar-Rum",        163),
    (31, "Luqman",             "Luqman",        165),
    (32, "Prostration",        "As-Sajdah",     167),
    (33, "The Confederates",   "Al-Ahzab",      168),
    (34, "Sheba",              "Saba",          172),
    (35, "Originator",         "Fatir",         174),
    (36, "Ya-Seen",            "Ya-Seen",       176),
    (37, "The Aligners",       "As-Saffat",     179),
    (38, "Saad",               "Saad",          183),
    (39, "Throngs",            "Az-Zumar",      185),
    (40, "Forgiver",           "Ghafir",        189),
    (41, "Detailed",           "Fussilat",      192),
    (42, "Consultation",       "Ash-Shura",     195),
    (43, "Decorations",        "Az-Zukhruf",    197),
    (44, "Smoke",              "Ad-Dukhan",     200),
    (45, "Kneeling",           "Al-Jathiyah",   201),
    (46, "The Dunes",          "Al-Ahqaf",      203),
    (47, "Muhammad",           "Muhammad",      205),
    (48, "Victory",            "Al-Fath",       207),
    (49, "The Chambers",       "Al-Hujurat",    208),
    (50, "Qaf",                "Qaf",           209),
    (51, "The Spreaders",      "Adh-Dhariyat",  211),
    (52, "The Mount",          "At-Tur",        212),
    (53, "The Star",           "An-Najm",       213),
    (54, "The Moon",           "Al-Qamar",      215),
    (55, "The Compassionate",  "Ar-Rahman",     216),
    (56, "The Inevitable",     "Al-Waqi'ah",    218),
    (57, "Iron",               "Al-Hadid",      220),
    (58, "The Argument",       "Al-Mujadilah",  221),
    (59, "The Mobilization",   "Al-Hashr",      223),
    (60, "The Woman Tested",   "Al-Mumtahinah", 224),
    (61, "Column",             "As-Saff",       225),
    (62, "Friday",             "Al-Jumu'ah",    226),
    (63, "The Hypocrites",     "Al-Munafiqun",  226),
    (64, "Gathering",          "At-Taghabun",   227),
    (65, "Divorce",            "At-Talaq",      228),
    (66, "Prohibition",        "At-Tahrim",     228),
    (67, "Sovereignty",        "Al-Mulk",       229),
    (68, "The Pen",            "Al-Qalam",      230),
    (69, "The Reality",        "Al-Haqqah",     232),
    (70, "Ways of Ascent",     "Al-Ma'arij",    233),
    (71, "Noah",               "Nuh",           234),
    (72, "The Jinn",           "Al-Jinn",       234),
    (73, "The Enwrapped",      "Al-Muzzammil",  235),
    (74, "The Enrobed",        "Al-Muddathir",  236),
    (75, "Resurrection",       "Al-Qiyamah",    237),
    (76, "Man",                "Al-Insan",      238),
    (77, "The Unleashed",      "Al-Mursalat",   239),
    (78, "The Event",          "An-Naba",       240),
    (79, "The Snatchers",      "An-Nazi'at",    241),
    (80, "He Frowned",         "Abasa",         241),
    (81, "The Rolling",        "At-Takwir",     242),
    (82, "The Shattering",     "Al-Infitar",    243),
    (83, "The Defrauders",     "Al-Mutaffifin", 243),
    (84, "The Rupture",        "Al-Inshiqaq",   244),
    (85, "The Constellations", "Al-Buruj",      244),
    (86, "The Nightly Visitor","At-Tariq",      245),
    (87, "The Most High",      "Al-A'la",       245),
    (88, "The Overwhelming",   "Al-Ghashiyah",  246),
    (89, "The Dawn",           "Al-Fajr",       246),
    (90, "The Land",           "Al-Balad",      247),
    (91, "The Sun",            "Ash-Shams",     247),
    (92, "The Night",          "Al-Layl",       248),
    (93, "Morning Light",      "Ad-Duha",       248),
    (94, "The Soothing",       "Ash-Sharh",     248),
    (95, "The Fig",            "At-Tin",        249),
    (96, "The Clot",           "Al-Alaq",       249),
    (97, "Decree",             "Al-Qadr",       249),
    (98, "Clear Evidence",     "Al-Bayyinah",   250),
    (99, "The Quake",          "Az-Zalzalah",   250),
    (100,"The Racers",         "Al-Adiyat",     250),
    (101,"The Shocker",        "Al-Qari'ah",    250),
    (102,"Rivalry",            "At-Takathur",   251),
    (103,"Time",               "Al-Asr",        251),
    (104,"The Backbiter",      "Al-Humazah",    251),
    (105,"The Elephant",       "Al-Fil",        251),
    (106,"Quraish",            "Quraish",       252),
    (107,"Assistance",         "Al-Ma'un",      252),
    (108,"Abundance",          "Al-Kawthar",    252),
    (109,"The Disbelievers",   "Al-Kafirun",    252),
    (110,"Victory",            "An-Nasr",       252),
    (111,"Thorns",             "Al-Masad",      252),
    (112,"Monotheism",         "Al-Ikhlas",     253),
    (113,"Daybreak",           "Al-Falaq",      253),
    (114,"Mankind",            "An-Nas",        253),
]

SURAH_OPTIONS = [f"{s[0]:3d}. {s[1]}  ({s[2]})" for s in SURAHS]


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QRAG — Quran Knowledge",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stChatMessage { border-radius: 12px; padding: 4px; }
.source-chip {
    display:inline-block; background:#1e293b; color:#94a3b8;
    border-radius:6px; padding:2px 8px; font-size:0.75rem; margin:2px;
}
.stApp { background-color: #0f172a; }
h1, h2, h3 { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource
def get_kb():
    return KnowledgeBase()

@st.cache_resource
def get_rag(_kb):
    return ReasoningRAG(_kb)

@st.cache_data
def get_pdf_page_count():
    if not QURAN_PDF.exists():
        return 0
    pdf = pymupdf.open(str(QURAN_PDF))
    n = len(pdf)
    pdf.close()
    return n

@st.cache_data(show_spinner=False)
def render_page(page_num: int) -> bytes:
    pdf  = pymupdf.open(str(QURAN_PDF))
    page = pdf[page_num - 1]
    mat  = pymupdf.Matrix(1.8, 1.8)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    data = pix.tobytes("png")
    pdf.close()
    return data


kb  = get_kb()
rag = get_rag(kb)
total_pdf_pages = get_pdf_page_count()

if "messages"      not in st.session_state: st.session_state.messages     = []
if "indexed_files" not in st.session_state: st.session_state.indexed_files = set(kb.list_sources())
if "reader_page"   not in st.session_state: st.session_state.reader_page  = 12


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📖 QRAG")
    st.caption("Quran Knowledge System")
    st.divider()

    st.subheader("📄 Index Documents")
    uploaded = st.file_uploader(
        "Drop PDFs here", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploaded:
        for uf in uploaded:
            if uf.name not in st.session_state.indexed_files:
                status_box   = st.empty()
                progress_bar = st.progress(0)

                def update_status(msg, _name=uf.name):
                    status_box.info(f"⏳ {msg}")
                    if "%" in msg:
                        try:
                            pct = int(msg.split(":")[1].strip().replace("%","")) / 100
                            progress_bar.progress(pct)
                        except Exception:
                            pass

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = tmp.name

                result = kb.index_pdf(tmp_path, progress_cb=update_status)
                os.unlink(tmp_path)

                if result["status"] == "indexed":
                    status_box.success(f"✅ {uf.name} — {result['chunks']} chunks")
                    st.session_state.indexed_files.add(uf.name)
                else:
                    status_box.info(f"⏭ {uf.name} already indexed")
                progress_bar.empty()

    st.divider()

    st.subheader("📚 Knowledge Base")
    total_chunks = kb.count()
    sources      = kb.list_sources()
    c1, c2 = st.columns(2)
    c1.metric("Chunks",    f"{total_chunks:,}")
    c2.metric("Documents", len(sources))

    if sources:
        with st.expander("Indexed documents", expanded=False):
            for src in sources:
                cs, cd = st.columns([4, 1])
                cs.caption(f"📄 {src}")
                if cd.button("🗑", key=f"del_{src}", help=f"Remove {src}"):
                    kb.delete_source(src)
                    st.session_state.indexed_files.discard(src)
                    st.rerun()

    st.divider()

    st.subheader("⚙️ Settings")
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        rag.clear_history()
        st.rerun()
    show_sources = st.toggle("Show source citations", value=True)

    st.divider()
    st.caption("LLM: **deepseek-r1:8b**")
    st.caption("Embeddings: **nomic-embed-text**")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_reader = st.tabs(["💬  Chat with Knowledge", "📖  Read Quran"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("Deep Reasoning Chat")
    if not sources:
        st.info("👈 Upload the Quran PDF (or any PDFs) in the sidebar to start chatting.")
    else:
        st.caption(f"Knowledge base · {total_chunks:,} chunks · {len(sources)} document(s)")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(
                            f'<span class="source-chip">📄 {s["source"]} · p.{s["page"]} · {s["score"]:.2f}</span>',
                            unsafe_allow_html=True,
                        )

    if prompt := st.chat_input(
        "Ask about the Quran, get guidance, explore any topic…",
        disabled=(total_chunks == 0),
    ):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_text   = ""
            for token in rag.chat(prompt, stream=True):
                full_text += token
                display = full_text
                if "<think>" in display:
                    display = display.replace(
                        "<think>",
                        '<details><summary style="color:#64748b;font-size:0.85rem">🤔 Reasoning…</summary>'
                        '<div style="color:#94a3b8;font-size:0.85rem">',
                    ).replace("</think>", "</div></details>")
                placeholder.markdown(display, unsafe_allow_html=True)

            srcs = rag.get_last_sources()
            if show_sources and srcs:
                with st.expander("📎 Sources", expanded=False):
                    for s in srcs:
                        st.markdown(
                            f'<span class="source-chip">📄 {s["source"]} · p.{s["page"]} · {s["score"]:.2f}</span>',
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text, "sources": srcs}
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QURAN READER
# ══════════════════════════════════════════════════════════════════════════════
with tab_reader:
    if not QURAN_PDF.exists():
        st.warning(f"Quran PDF not found at `{QURAN_PDF}`. Place it in `PDF_Data/` to enable the reader.")
    else:
        st.header("Quran Reader")

        # ── Controls ─────────────────────────────────────────────────────────
        col_surah, col_page, col_nav = st.columns([4, 2, 2])

        with col_surah:
            sel = st.selectbox(
                "Jump to Surah",
                options=range(len(SURAH_OPTIONS)),
                format_func=lambda i: SURAH_OPTIONS[i],
                label_visibility="visible",
            )
            if st.button("↗ Go to this Surah", use_container_width=True):
                st.session_state.reader_page = SURAHS[sel][3]
                st.rerun()

        with col_page:
            st.markdown("<br>", unsafe_allow_html=True)
            new_page = st.number_input(
                f"Page (1–{total_pdf_pages})",
                min_value=1,
                max_value=total_pdf_pages,
                value=st.session_state.reader_page,
                step=1,
            )
            if new_page != st.session_state.reader_page:
                st.session_state.reader_page = int(new_page)
                st.rerun()

        with col_nav:
            st.markdown("<br>", unsafe_allow_html=True)
            n1, n2 = st.columns(2)
            if n1.button("◀ Prev", use_container_width=True):
                st.session_state.reader_page = max(1, st.session_state.reader_page - 1)
                st.rerun()
            if n2.button("Next ▶", use_container_width=True):
                st.session_state.reader_page = min(total_pdf_pages, st.session_state.reader_page + 1)
                st.rerun()

        st.divider()

        # ── Current surah label ───────────────────────────────────────────────
        cur = st.session_state.reader_page
        current_surah = next(
            (s for s in reversed(SURAHS) if s[3] <= cur), None
        )
        if current_surah:
            st.markdown(
                f"**Page {cur} / {total_pdf_pages}** &nbsp;·&nbsp; "
                f"Surah {current_surah[0]}: *{current_surah[1]}* ({current_surah[2]})",
                unsafe_allow_html=True,
            )

        # ── Render page ───────────────────────────────────────────────────────
        with st.spinner("Rendering…"):
            img_bytes = render_page(cur)

        img_b64 = base64.b64encode(img_bytes).decode()
        st.markdown(
            f'<div style="display:flex;justify-content:center;background:#fdf8f0;'
            f'border-radius:10px;padding:16px;margin-top:8px;">'
            f'<img src="data:image/png;base64,{img_b64}" '
            f'style="max-width:760px;width:100%;border-radius:6px;'
            f'box-shadow:0 6px 24px rgba(0,0,0,0.35);" /></div>',
            unsafe_allow_html=True,
        )

        # ── Bottom nav ────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1, 2, 1])
        if b1.button("◀ Previous page", use_container_width=True, key="b_prev"):
            st.session_state.reader_page = max(1, cur - 1)
            st.rerun()
        b2.markdown(
            f"<p style='text-align:center;color:#94a3b8;margin-top:8px'>"
            f"Page {cur} of {total_pdf_pages}</p>",
            unsafe_allow_html=True,
        )
        if b3.button("Next page ▶", use_container_width=True, key="b_next"):
            st.session_state.reader_page = min(total_pdf_pages, cur + 1)
            st.rerun()
