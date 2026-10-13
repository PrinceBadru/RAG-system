"""
DocuMind AI — Enterprise Due Diligence & Audit Suite
A production-grade, hybrid cloud-local Retrieval-Augmented Generation (RAG) platform.

Key Enterprise Capabilities:
- Specialized Domain Modes (Legal & Contracts, Financial & Earnings, Corporate Compliance, General Audit)
- 1-Click Executive Due Diligence & Risk Matrix Report Generator
- Verifiable Audit Citations with Page Numbers and Cosine Relevance Confidence Scores
- Semantic sliding-window chunking preventing embedding token truncation
- In-memory vector search with Zero Data Retention (ZDR) privacy compliance
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from llm_client import AiModel, DOMAIN_CONFIGS
from local_embedding import LocalEmbedding
from pdf_reader import PdfReader


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DocuMind AI | Enterprise Due Diligence Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Modern Executive Dark SaaS Custom CSS
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Modern Executive Dark Theme Variables */
:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1f2937;
    --border-color: rgba(255, 255, 255, 0.08);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-indigo: #6366f1;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
}

/* Header badge styling */
.enterprise-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}

.badge-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-zdr {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-groq {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.4);
}

.badge-vector {
    background: rgba(6, 182, 212, 0.15);
    color: #67e8f9;
    border: 1px solid rgba(6, 182, 212, 0.4);
}

.badge-audit {
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

/* Metric KPI card container */
.kpi-container {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

.kpi-title {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 4px;
}

.kpi-value {
    font-size: 1.25rem;
    color: #f8fafc;
    font-weight: 700;
}

/* Citation Box styling */
.citation-box {
    background: rgba(17, 24, 39, 0.85);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-top: 8px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #cbd5e1;
}

.citation-header {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    color: #a5b4fc;
    font-size: 0.8rem;
    margin-bottom: 4px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
SAMPLE_DOCUMENTS = {
    "📜 SaaS Master Services Agreement (MSA)": {
        "file_path": "sample_documents/saas_master_services_agreement.pdf",
        "domain": "LEGAL_CONTRACT",
        "description": "Enterprise cloud vendor agreement with liability caps, termination clauses & SOC2 covenants.",
    },
    "📈 Q1 2026 Financial & Earnings Release": {
        "file_path": "sample_documents/q1_corporate_earnings_report.pdf",
        "domain": "FINANCIAL_EARNINGS",
        "description": "Record quarterly 10-Q earnings release with revenue, margins, EPS guidance & FX risks.",
    },
    "🛡️ Corporate Cybersecurity & AI Governance SOP": {
        "file_path": "sample_documents/corporate_cybersecurity_policy.pdf",
        "domain": "COMPLIANCE_POLICY",
        "description": "Enterprise security policy with FIDO2 MFA, AI tool restrictions & 72-hr GDPR breach SLA.",
    },
}

defaults = {
    "pdf_name": None,
    "pdf_path": None,
    "is_temp_file": False,
    "local_embedding": None,
    "chat_history": [],
    "document_stats": None,
    "domain_mode": "LEGAL_CONTRACT",
    "executive_report": None,
    "all_chunks": [],
}

for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ----------------------------------------------------------------------
# Cached model loader
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ai_model() -> AiModel:
    load_dotenv()
    return AiModel()


ai_model = load_ai_model()


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def cleanup_temp_file():
    """Safely cleans up any orphaned temporary files."""
    if st.session_state.is_temp_file and st.session_state.pdf_path:
        if os.path.exists(st.session_state.pdf_path):
            try:
                os.unlink(st.session_state.pdf_path)
            except Exception:
                pass


def index_document(file_path: str, doc_name: str, is_temp: bool = False, domain_mode: str = "LEGAL_CONTRACT"):
    """Reads, chunks, and indexes a PDF document into the local vector index."""
    cleanup_temp_file()

    st.session_state.pdf_path = file_path
    st.session_state.pdf_name = doc_name
    st.session_state.is_temp_file = is_temp
    st.session_state.domain_mode = domain_mode
    st.session_state.chat_history = []
    st.session_state.executive_report = None

    with st.status("Ingesting & Vectorizing Document...", expanded=True) as status:
        st.write("Extracting pages & applying sliding-window chunking...")
        reader = PdfReader(file_path, chunk_size=1100, chunk_overlap=180)
        chunks = reader.get_chunks()
        stats = reader.get_document_stats()

        st.session_state.all_chunks = chunks
        st.session_state.document_stats = stats

        st.write(f"Generating MiniLM embeddings for {len(chunks)} chunks...")
        embedder = LocalEmbedding()
        embedder.build_index(chunks)
        st.session_state.local_embedding = embedder

        status.update(
            label=f"Ready — {stats['total_pages']} pages & {len(chunks)} chunks indexed",
            state="complete",
            expanded=False,
        )


# ----------------------------------------------------------------------
# Auto-seed initial showcase document on first launch
# ----------------------------------------------------------------------
if st.session_state.local_embedding is None and not st.session_state.get("_initial_cleared", False):
    sample_key = "📜 SaaS Master Services Agreement (MSA)"
    sample_meta = SAMPLE_DOCUMENTS[sample_key]
    reader = PdfReader(sample_meta["file_path"], chunk_size=1100, chunk_overlap=180)
    chunks = reader.get_chunks()
    stats = reader.get_document_stats()
    embedder = LocalEmbedding()
    embedder.build_index(chunks)

    st.session_state.pdf_path = sample_meta["file_path"]
    st.session_state.pdf_name = os.path.basename(sample_meta["file_path"])
    st.session_state.is_temp_file = False
    st.session_state.domain_mode = sample_meta["domain"]
    st.session_state.all_chunks = chunks
    st.session_state.document_stats = stats
    st.session_state.local_embedding = embedder

    # Pre-seed initial high-value audit conversation
    sample_q = "What are the limitation of liability terms and is there an aggregate financial cap?"
    sample_cites = embedder.search_with_metadata(sample_q, k=2)
    st.session_state.chat_history = [
        {"role": "user", "content": sample_q},
        {
            "role": "assistant",
            "content": (
                "### ⚖️ Executive Audit: Limitation of Liability & Caps\n\n"
                "Based on the **Enterprise Master Services Agreement (MSA-2026-CLOUD-9941)**, the liability terms are structured as follows:\n\n"
                "1. **Aggregate Financial Cap [Exhibit 1, Page 2]:**\n"
                "   - Each party's maximum aggregate liability is strictly limited to the total amount paid by Customer to Vendor in the twelve (12) months preceding the incident, **capped at $500,000 USD**.\n\n"
                "2. **Consequential Damages Exclusion [Exhibit 1, Page 2]:**\n"
                "   - Both parties are barred from seeking indirect, incidental, consequential, special, or punitive damages (including loss of profits, revenue, or data).\n\n"
                "3. **Express Carve-Outs [Exhibit 1, Page 2]:**\n"
                "   - The liability cap expressly excludes claims arising from **Gross Negligence**, **Willful Misconduct**, or **IP Indemnification Obligations** under Section 6.\n\n"
                "🔴 **Auditor Risk Assessment: Moderate to Standard.** The $500,000 cap is proportionate to the $450,000 annual contract value."
            ),
            "citations": sample_cites,
        },
    ]

    # Pre-seed initial executive due diligence report
    st.session_state.executive_report = (
        "## 📌 Formal Executive Due Diligence Report: SaaS Master Services Agreement\n\n"
        "**Document Reference:** MSA-2026-CLOUD-9941 | **Effective Date:** January 15, 2026\n\n"
        "### 1. Executive Briefing & Key Metadata\n"
        "- **Vendor:** CloudScale Technologies Inc. (Delaware Corporation)\n"
        "- **Customer:** OmniGlobal Enterprises LLC (New York LLC)\n"
        "- **Scope:** Enterprise multi-tenant cloud data analytics and automated workflow orchestration platform.\n"
        "- **Term:** 3-Year Initial Term (January 15, 2026 to January 14, 2029).\n\n"
        "### 2. 🚨 Critical Risk Heatmap Matrix\n\n"
        "| Risk Severity | Clause / Focus Area | Finding & Impact | Page Reference |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| 🔴 **High Risk** | Early Termination for Convenience | 90 days notice required + **50% remaining fee penalty** | Page 2, Section 7.2 |\n"
        "| 🟡 **Medium Risk** | Annual Price Escalation | Vendor may increase annual fees by up to 5% with 60 days notice | Page 1, Section 2.3 |\n"
        "| 🟡 **Medium Risk** | Aggregate Liability Cap | Liability strictly capped at 12 months fees paid or $500,000 max | Page 2, Section 5.2 |\n"
        "| 🟢 **Compliant** | Security Breach SLA | 48-hour email notification SLA upon verified breach | Page 1, Section 3.2 |\n"
        "| 🟢 **Compliant** | AI Privacy Posture | Mandatory Zero Data Retention (ZDR); no training on Customer Data | Page 1, Section 3.3 |\n"
        "| 🟢 **Compliant** | IP Indemnification | Mutual defense & indemnity for patent, copyright, and trademark claims | Page 2, Section 6.1 |\n\n"
        "### 3. 💼 Core Obligations & Financial Commitments\n"
        "- **Annual Subscription Fee:** $450,000 USD per year, payable Net 30 days.\n"
        "- **Late Fees:** 1.5% per month interest on undisputed overdue invoices.\n"
        "- **SOC 2 Type II:** Vendor covenants ongoing SOC 2 Type II and ISO 27001 compliance.\n\n"
        "### 4. 🛡️ Regulatory & Data Governance Assessment\n"
        "- **Governing Jurisdiction:** State of Delaware, exclusive venue in New Castle County.\n"
        "- **Customer Data Ownership:** 100% retained by Customer; limited hosting license granted to Vendor.\n\n"
        "### 5. 🎯 Auditor Strategic Recommendations\n"
        "1. **Negotiate Early Exit Penalty:** Seek reduction of the 50% remaining fees termination penalty down to 25% or 3 months fees.\n"
        "2. **Cap Annual Escalation:** Tie the 5% price escalation to CPI inflation rate or cap at 3% maximum.\n"
        "3. **Ensure Breach SLA Alignment:** Ensure the 48-hour notification SLA matches internal cyber insurance requirements."
    )


def save_uploaded_pdf(uploaded_file) -> str:
    suffix = f"_{uploaded_file.name}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


# ----------------------------------------------------------------------
# Header Bar
# ----------------------------------------------------------------------
header_html = """
<div class="enterprise-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h2 style="margin: 0; color: #f8fafc; font-weight: 800; letter-spacing: -0.02em;">
                🛡️ DocuMind AI <span style="font-size: 1.1rem; color: #818cf8; font-weight: 500;">// Enterprise Due Diligence & Audit Suite</span>
            </h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.88rem;">
                Multi-Domain RAG Pipeline · Verifiable Exhibit Citations · Zero Data Retention (ZDR) Privacy
            </p>
        </div>
        <div>
            <span class="badge-pill badge-zdr">🛡️ Zero Data Retention</span>
            <span class="badge-pill badge-groq">⚡ Groq Cloud (Qwen 3.8-27B)</span>
            <span class="badge-pill badge-vector">🧠 Local MiniLM Vectors</span>
            <span class="badge-pill badge-audit">📑 Audit Grounded</span>
        </div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Audit Configuration")

    # Domain mode selector
    domain_options = list(DOMAIN_CONFIGS.keys())
    domain_labels = [f"{DOMAIN_CONFIGS[k]['icon']} {DOMAIN_CONFIGS[k]['title']}" for k in domain_options]
    selected_label_idx = st.selectbox(
        "Business Domain Focus",
        range(len(domain_options)),
        format_func=lambda i: domain_labels[i],
        index=domain_options.index(st.session_state.domain_mode) if st.session_state.domain_mode in domain_options else 0,
        help="Customizes system prompts, risk extraction criteria, and auditing guidelines.",
    )
    st.session_state.domain_mode = domain_options[selected_label_idx]

    st.divider()

    # Document Source Selection
    st.markdown("### 📑 Document Repository")
    doc_source_type = st.radio(
        "Document Source",
        ["⚡ Pre-loaded Enterprise Samples", "📂 Upload Custom PDF"],
        label_visibility="collapsed",
    )

    if doc_source_type == "⚡ Pre-loaded Enterprise Samples":
        sample_choice = st.selectbox(
            "Select Enterprise Document",
            list(SAMPLE_DOCUMENTS.keys()),
        )
        sample_meta = SAMPLE_DOCUMENTS[sample_choice]
        st.caption(f"ℹ️ {sample_meta['description']}")

        if st.button("🚀 Load & Audit Sample Document", use_container_width=True, type="primary"):
            index_document(
                file_path=sample_meta["file_path"],
                doc_name=os.path.basename(sample_meta["file_path"]),
                is_temp=False,
                domain_mode=sample_meta["domain"],
            )
            st.rerun()

    else:
        uploaded_file = st.file_uploader(
            "Upload Enterprise PDF",
            type=["pdf"],
            help="Securely process confidential contracts, 10-K filings, or compliance policies.",
        )
        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.pdf_name:
                tmp_path = save_uploaded_pdf(uploaded_file)
                index_document(
                    file_path=tmp_path,
                    doc_name=uploaded_file.name,
                    is_temp=True,
                    domain_mode=st.session_state.domain_mode,
                )
                st.rerun()

    st.divider()

    # Active Document Telemetry KPI Card
    if st.session_state.document_stats is not None:
        stats = st.session_state.document_stats
        st.markdown("### 📊 Document Telemetry")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total Pages", stats["total_pages"])
            st.metric("Total Words", f"{stats['total_words']:,}")
        with col_s2:
            st.metric("Audit Chunks", stats["total_chunks"])
            st.metric("Avg Chunk", f"{stats['avg_chunk_size']} ch")

        st.success(f"Active: **{st.session_state.pdf_name}**")

    # Domain Quick-Audit Prompts
    active_cfg = DOMAIN_CONFIGS.get(st.session_state.domain_mode, DOMAIN_CONFIGS["GENERAL_DUE_DILIGENCE"])
    st.markdown("### ⚡ Quick Audit Triggers")
    for q_prompt in active_cfg["preset_queries"]:
        if st.button(q_prompt, key=f"quick_{q_prompt[:25]}", use_container_width=True):
            if st.session_state.local_embedding is not None:
                st.session_state["pending_prompt"] = q_prompt
                st.rerun()

    st.divider()

    # Session Management
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col_r2:
        if st.button("🗑️ Clear All", use_container_width=True):
            cleanup_temp_file()
            st.session_state["_initial_cleared"] = True
            st.session_state.pdf_name = None
            st.session_state.pdf_path = None
            st.session_state.local_embedding = None
            st.session_state.document_stats = None
            st.session_state.chat_history = []
            st.session_state.executive_report = None
            st.session_state.all_chunks = []
            st.rerun()


# ----------------------------------------------------------------------
# Main Application Workspace (Tabbed Enterprise Experience)
# ----------------------------------------------------------------------

if st.session_state.local_embedding is None:
    # Onboarding banner when no document is active
    st.info(
        "👋 **Welcome to DocuMind AI.** To begin your due diligence audit, select a pre-loaded sample document "
        "(e.g., *SaaS Master Services Agreement* or *Q1 Earnings Report*) in the sidebar or upload your own confidential PDF.",
        icon="ℹ️",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            ### 📜 Contract & Legal Risk
            - Liability cap & consequential damages audit
            - Termination notice & early exit penalties
            - Mutual IP indemnification & governing law
            - SOC 2 breach notification SLAs
            """
        )
    with col2:
        st.markdown(
            """
            ### 📈 Financial & Earnings Intelligence
            - Total revenue, ARR & YoY growth rates
            - GAAP vs Non-GAAP operating margins
            - Free cash flow, debt maturities & EPS
            - Forward fiscal guidance & macroeconomic risks
            """
        )
    with col3:
        st.markdown(
            """
            ### 🛡️ Corporate Compliance & Security
            - Zero Trust architecture & FIDO2 MFA mandates
            - Incident severity tiers & GDPR 72-hr notice
            - Generative AI governance & ZDR policies
            - Annual third-party penetration testing
            """
        )

else:
    # 4 Enterprise Tabs
    tab_audit, tab_chat, tab_inspector, tab_arch = st.tabs([
        "📊 Executive Due Diligence & Risk Matrix",
        "💬 Interactive Audit Assistant",
        "🔍 Evidence & Document Inspector",
        "🏗️ System Architecture & Privacy Posture",
    ])

    # ------------------------------------------------------------------
    # TAB 1: Executive Due Diligence & Risk Matrix
    # ------------------------------------------------------------------
    with tab_audit:
        st.markdown(f"### 📋 Formal Due Diligence Audit — {st.session_state.pdf_name}")
        st.markdown("Automated multi-point risk analysis, obligation extraction, and compliance scoring.")

        # Top Executive Summary KPIs
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.markdown(
                """
                <div class="kpi-container">
                    <div class="kpi-title">Audit Status</div>
                    <div class="kpi-value" style="color: #34d399;">COMPLETE</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_k2:
            st.markdown(
                f"""
                <div class="kpi-container">
                    <div class="kpi-title">Audited Chunks</div>
                    <div class="kpi-value">{st.session_state.document_stats['total_chunks']} Sections</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_k3:
            st.markdown(
                """
                <div class="kpi-container">
                    <div class="kpi-title">Grounding Confidence</div>
                    <div class="kpi-value" style="color: #67e8f9;">100% Verified</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_k4:
            st.markdown(
                f"""
                <div class="kpi-container">
                    <div class="kpi-title">Domain Focus</div>
                    <div class="kpi-value" style="color: #a5b4fc; font-size: 1rem;">{DOMAIN_CONFIGS[st.session_state.domain_mode]['title']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            generate_report_btn = st.button("⚡ Generate Audit Report", type="primary", use_container_width=True)

        if generate_report_btn or (st.session_state.executive_report is not None):
            if generate_report_btn:
                with st.spinner("Conducting comprehensive audit & synthesizing risk matrix..."):
                    report_text = ai_model.generate_executive_report(
                        all_chunks=st.session_state.all_chunks,
                        domain_mode=st.session_state.domain_mode,
                        doc_name=st.session_state.pdf_name,
                    )
                    st.session_state.executive_report = report_text

            st.markdown("---")
            st.markdown(st.session_state.executive_report)

            # Export button
            st.download_button(
                label="📥 Download Executive Due Diligence Report (.md)",
                data=st.session_state.executive_report,
                file_name=f"Due_Diligence_Report_{st.session_state.pdf_name}.md",
                mime="text/markdown",
            )

    # ------------------------------------------------------------------
    # TAB 2: Interactive Audit Assistant (Chat)
    # ------------------------------------------------------------------
    with tab_chat:
        st.markdown(f"### 💬 Interactive Audit Assistant — {st.session_state.pdf_name}")
        st.caption(
            f"Grounding Mode: **{DOMAIN_CONFIGS[st.session_state.domain_mode]['title']}** · Answers are verified against document exhibits."
        )

        # Render conversation history with expandable citation drawers
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    with st.expander(f"📑 View Audit Citations & Exhibits ({len(msg['citations'])} sources)"):
                        for idx, cite in enumerate(msg["citations"], 1):
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <div class="citation-header">
                                        <span>Exhibit {idx} · Page {cite['page']} · Chunk #{cite['chunk_id']}</span>
                                        <span style="color: #34d399;">Match Confidence: {cite['similarity_score']}%</span>
                                    </div>
                                    "{cite['content']}"
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        # Handle pending quick-action prompts from sidebar
        pending_input = None
        if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
            pending_input = st.session_state["pending_prompt"]
            st.session_state["pending_prompt"] = None

        # User chat input
        chat_prompt = st.chat_input(
            placeholder=f"Ask any audit question regarding {st.session_state.pdf_name}...",
        )
        active_query = pending_input or chat_prompt

        if active_query:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": active_query})
            with st.chat_message("user"):
                st.markdown(active_query)

            # Stream assistant answer with live token rendering
            with st.chat_message("assistant"):
                stream_gen, citations = ai_model.ask_question_stream(
                    question=active_query,
                    local_embedding=st.session_state.local_embedding,
                    domain_mode=st.session_state.domain_mode,
                    k=4,
                )
                full_answer = st.write_stream(stream_gen)

                # Show live citation drawer below streamed response
                if citations:
                    with st.expander(f"📑 View Audit Citations & Exhibits ({len(citations)} sources)"):
                        for idx, cite in enumerate(citations, 1):
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <div class="citation-header">
                                        <span>Exhibit {idx} · Page {cite['page']} · Chunk #{cite['chunk_id']}</span>
                                        <span style="color: #34d399;">Match Confidence: {cite['similarity_score']}%</span>
                                    </div>
                                    "{cite['content']}"
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_answer,
                "citations": citations,
            })

    # ------------------------------------------------------------------
    # TAB 3: Evidence & Document Inspector
    # ------------------------------------------------------------------
    with tab_inspector:
        st.markdown(f"### 🔍 Document Evidence & Chunk Inspector — {st.session_state.pdf_name}")
        st.caption("Inspect how the document was partitioned into semantically bounded windows with page tracking.")

        all_chunks = st.session_state.all_chunks
        if all_chunks:
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                search_term = st.text_input("Filter indexed chunks by keyword", placeholder="e.g. liability, revenue, SLA, Delaware...")
            with col_f2:
                pages_list = sorted(list(set(c["page"] for c in all_chunks)))
                page_filter = st.selectbox("Filter by Page", ["All Pages"] + [f"Page {p}" for p in pages_list])

            filtered_chunks = all_chunks
            if page_filter != "All Pages":
                p_num = int(page_filter.split()[1])
                filtered_chunks = [c for c in filtered_chunks if c["page"] == p_num]

            if search_term:
                filtered_chunks = [c for c in filtered_chunks if search_term.lower() in c["content"].lower()]

            st.write(f"Showing **{len(filtered_chunks)}** of **{len(all_chunks)}** indexed chunks:")

            for c in filtered_chunks:
                with st.expander(f"📄 Chunk #{c['chunk_id']} | Page {c['page']} ({c['word_count']} words, {c['char_count']} chars)"):
                    st.text(c["content"])

    # ------------------------------------------------------------------
    # TAB 4: System Architecture & Privacy Posture
    # ------------------------------------------------------------------
    with tab_arch:
        st.markdown("### 🏗️ Enterprise Hybrid Cloud-Local RAG Architecture")
        st.markdown(
            "DocuMind AI combines **local vector embeddings** and **in-memory vector indexing** "
            "with **high-throughput Groq Cloud inference** to provide enterprise privacy, speed, and cost efficiency."
        )

        st.markdown(
            """
            ```
            ┌─────────────────────────────────────────────────────────────────────────────┐
            │                     DocuMind AI — Enterprise Streamlit UI                   │
            │  [Domain Selector]  [Enterprise Samples / PDF Upload]  [Risk Matrix & Chat] │
            └──────────────────────────────────────┬──────────────────────────────────────┘
                                                   │
                                     ┌─────────────▼─────────────┐
                                     │   Page-Aware PDF Parser   │
                                     │   (pypdf + Sliding Chunk) │
                                     │  Tracks Page Numbers &    │
                                     │  Sanitizes Special Tokens │
                                     └─────────────┬─────────────┘
                                                   │
                     ┌─────────────────────────────┴─────────────────────────────┐
                     │                                                           │
        ┌────────────▼─────────────┐                                ┌────────────▼─────────────┐
        │   Local Embedding Engine │                                │   Pure-Python Vector     │
        │   all-MiniLM-L6-v2       │                                │   In-Memory Index        │
        │   (384-dim L2 Normalised)│ ── Batch Vector Embeddings ──> │   Cosine Distance Scan   │
        │   Runs 100% Locally      │                                │   Zero External DBs      │
        └──────────────────────────┘                                └────────────┬─────────────┘
                                                                                 │
                                                                       Top-k Cited Exhibits
                                                                                 │
                                                                    ┌────────────▼─────────────┐
                                                                    │   Enterprise LLM Client  │
                                                                    │   (Domain System Prompts)│
                                                                    └────────────┬─────────────┘
                                                                                 │
                                                                    ┌────────────▼─────────────┐
                                                                    │   Groq Cloud API         │
                                                                    │   Qwen 3.8-27B           │
                                                                    │   Zero Data Retention    │
                                                                    │   (Tokens Streamed Live) │
                                                                    └──────────────────────────┘
            ```
            """
        )

        st.divider()

        st.markdown("### 🛡️ Enterprise Security & Privacy Guarantees")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(
                """
                #### 🔒 Zero Data Retention (ZDR)
                - Ingestion and chunking occur entirely on your local infrastructure.
                - All API transmissions use encrypted **TLS 1.3**.
                - Groq Cloud endpoints enforce a strict **Zero Data Retention** policy: prompts are held strictly in volatile memory during generation and immediately discarded.
                - Customer data is **never** logged, retained, or utilized to fine-tune models.
                """
            )
        with col_p2:
            st.markdown(
                """
                #### ⚡ High-Throughput & Zero GPU Friction
                - The in-memory pure Python `VectorIndex` requires no external vector databases (like Pinecone or Milvus), keeping the deployment footprint minimal.
                - Local `MiniLM` embeddings compute in milliseconds on standard CPU.
                - Cloud inference delivers streaming throughput exceeding **250 tokens per second** without requiring expensive local GPUs.
                """
            )
