"""
Enterprise Due Diligence & Audit LLM Engine.
Powered by Groq Cloud API with Zero Data Retention (ZDR) privacy compliance.

Features:
- Domain-specific system prompts (Legal & Contract, Financial & Earnings, Corporate Compliance, General Audit)
- Strict grounding with audit exhibit citations
- Automated 1-Click Executive Due Diligence & Risk Matrix generator
- Real-time token streaming with citation tracking
- Prompt injection and special token sanitization
"""

import os
import re
from typing import Generator, Tuple, List, Dict, Any
from groq import Groq
from local_embedding import LocalEmbedding


DOMAIN_CONFIGS = {
    "LEGAL_CONTRACT": {
        "title": "Legal & Contract Risk Analysis",
        "icon": "⚖️",
        "persona": (
            "You are a Senior Corporate Legal Counsel and Due Diligence Auditor. "
            "Your job is to critically analyze commercial contracts, NDAs, and Master Services Agreements (MSAs). "
            "Focus on: Limitation of liability caps, consequential damages exclusions, indemnification obligations, "
            "termination for convenience penalties, breach notification SLAs, IP assignment, and governing jurisdiction. "
            "Always classify contractual risk rigorously (High Risk 🔴, Medium Risk 🟡, Compliant 🟢)."
        ),
        "preset_queries": [
            "What are the limitation of liability terms and is there an aggregate financial cap?",
            "What are the termination clauses, notice periods, and early cancellation penalties?",
            "What data security covenants, breach notification SLAs, and certifications are required?",
            "Analyze the indemnification obligations and identify any non-standard risks.",
        ],
    },
    "FINANCIAL_EARNINGS": {
        "title": "Financial & Earnings Intelligence",
        "icon": "📈",
        "persona": (
            "You are a Principal Financial Analyst and Corporate M&A Due Diligence Specialist. "
            "Your job is to examine quarterly earnings releases, 10-K/10-Q reports, and financial statements. "
            "Focus on: Revenue growth (YoY/QoQ), ARR, GAAP vs Non-GAAP operating income, gross margins, EBITDA, "
            "free cash flow, diluted EPS, forward fiscal guidance, debt maturities, and macroeconomic headwinds. "
            "Be exact with all numerical metrics and state exact percentage variations."
        ),
        "preset_queries": [
            "Summarize Q1 revenue, YoY growth percentage, and compare against consensus expectations.",
            "What is the full-year forward fiscal guidance for revenue, operating margin, and EPS?",
            "What is the balance sheet liquidity, cash position, and long-term debt profile?",
            "What macroeconomic headwinds, FX exposures, or margin risks were disclosed?",
        ],
    },
    "COMPLIANCE_POLICY": {
        "title": "Cybersecurity & Policy Compliance",
        "icon": "🛡️",
        "persona": (
            "You are a Chief Information Security Officer (CISO) and Regulatory Compliance Auditor. "
            "Your job is to evaluate enterprise security SOPs, AI governance frameworks, and data protection policies. "
            "Focus on: Authentication standards (MFA/FIDO2), Zero Trust architecture, incident severity tiers (P1/P2/P3), "
            "regulatory notification timelines (GDPR 72-hour SLAs), AI tool restrictions (Zero Data Retention), "
            "and SOC 2 / ISO 27001 audit cadences."
        ),
        "preset_queries": [
            "What are the incident severity classifications and regulatory notification SLAs?",
            "What are the corporate policies and restrictions regarding generative AI and LLM usage?",
            "What identity, MFA, and Zero Trust access control requirements are mandated?",
            "What are the audit cadences, penetration testing frequencies, and non-compliance penalties?",
        ],
    },
    "GENERAL_DUE_DILIGENCE": {
        "title": "General Enterprise Due Diligence",
        "icon": "🔍",
        "persona": (
            "You are an Executive Enterprise Due Diligence Auditor. "
            "Your job is to perform a thorough, objective audit of corporate documentation. "
            "Focus on factual accuracy, identifying operational risks, noting governing terms, "
            "and delivering clear executive takeaways."
        ),
        "preset_queries": [
            "Provide an executive briefing of the primary commitments and parties in this document.",
            "Identify the top 3 highest-risk provisions or liabilities.",
            "What operational milestones, deadlines, and deliverables are specified?",
            "What governing law, dispute resolution, or compliance obligations apply?",
        ],
    },
}


class AiModel:
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        """
        Initializes the Groq Cloud API client for lightning-fast inference.
        """
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            print("WARNING: GROQ_API_KEY not found in environment variables. Groq API calls will fail.")

        print(f"[AiModel] Connecting to Groq Cloud API with model: {self.model_name}")
        self.client = Groq(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Sanitization and message construction
    # ------------------------------------------------------------------

    def _sanitize_input(self, text: str) -> str:
        """
        Strips potential model control markers to prevent prompt injection.
        """
        disallowed = ["<|end|>", "<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"]
        for marker in disallowed:
            text = text.replace(marker, "")
        return text.strip()

    def build_audit_messages(self, context_str: str, question: str, domain_mode: str) -> List[Dict[str, str]]:
        """
        Constructs rigorously grounded system and user prompts with domain specialization.
        """
        cfg = DOMAIN_CONFIGS.get(domain_mode, DOMAIN_CONFIGS["GENERAL_DUE_DILIGENCE"])
        persona = cfg["persona"]

        system_prompt = (
            f"{persona}\n\n"
            "STRICT AUDIT GUIDELINES:\n"
            "1. Base your answer EXCLUSIVELY on the provided Document Exhibits below. Do NOT assume, extrapolate, or use external knowledge.\n"
            "2. If an answer cannot be verified from the exhibits, explicitly state: 'The document does not contain verifiable information on this topic.'\n"
            "3. Cite your sources using exact inline references like '[Exhibit 1, Page X]' whenever quoting terms, numbers, or clauses.\n"
            "4. Structure your response with clear executive headers, bullet points, and highlight risk levels (🔴 High Risk, 🟡 Moderate, 🟢 Compliant) where appropriate.\n\n"
            "VERIFIED DOCUMENT EXHIBITS:\n"
            f"{context_str}\n"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"AUDIT INQUIRY: {self._sanitize_input(question)}"},
        ]

    # ------------------------------------------------------------------
    # Streaming Q&A
    # ------------------------------------------------------------------

    def ask_question_stream(
        self,
        question: str,
        local_embedding: LocalEmbedding,
        domain_mode: str = "LEGAL_CONTRACT",
        k: int = 4,
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """
        Retrieves context with citation metadata and streams the token response.

        Returns:
            Tuple of:
            - Token generator for Streamlit `st.write_stream`
            - List of citation metadata dictionaries
        """
        context_str, citations = local_embedding.get_context_with_citations(question, k=k)
        messages = self.build_audit_messages(context_str, question, domain_mode)

        stream = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=1500,
            temperature=0.2,  # Low temperature for strict factual grounding
            stream=True,
        )

        def token_generator():
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content

        return token_generator(), citations

    # ------------------------------------------------------------------
    # Automated 1-Click Executive Due Diligence Report Generator
    # ------------------------------------------------------------------

    def generate_executive_report(
        self,
        all_chunks: List[Dict[str, Any]],
        domain_mode: str = "LEGAL_CONTRACT",
        doc_name: str = "Document",
    ) -> str:
        """
        Performs an automated end-to-end due diligence audit across the entire document
        and generates an executive-ready risk matrix and briefing.
        """
        cfg = DOMAIN_CONFIGS.get(domain_mode, DOMAIN_CONFIGS["GENERAL_DUE_DILIGENCE"])
        persona = cfg["persona"]

        # Concatenate key chunks into an audit package (up to ~12k characters to fit context comfortably)
        selected_text_blocks = []
        char_count = 0
        for chunk in all_chunks:
            block = f"[Page {chunk['page']}, Chunk {chunk['chunk_id']}]:\n{chunk['content']}"
            if char_count + len(block) > 16000:
                break
            selected_text_blocks.append(block)
            char_count += len(block)

        combined_doc_context = "\n\n---\n\n".join(selected_text_blocks)

        system_instruction = (
            f"{persona}\n\n"
            f"You are conducting a formal Executive Due Diligence Audit on the document: '{doc_name}'.\n"
            "Generate a comprehensive, executive-grade Due Diligence Report structured strictly into these 5 sections:\n\n"
            "### 1. 📌 Executive Briefing & Key Metadata\n"
            "- Document Identity & Classification\n"
            "- Core Parties / Entities / Subject Matter\n"
            "- Strategic Purpose & High-Level Summary\n\n"
            "### 2. 🚨 Critical Risk Heatmap Matrix\n"
            "Create a Markdown Table with the following columns:\n"
            "| Risk Severity | Clause / Focus Area | Finding & Impact | Page Reference |\n"
            "Classify every key issue as:\n"
            "- 🔴 High Risk (e.g. Uncapped liability, severe penalty, short termination notice, non-standard indemnity, heavy debt, security gap)\n"
            "- 🟡 Medium Risk (e.g. Annual price escalation, currency exposure, JIT access limits)\n"
            "- 🟢 Low Risk / Compliant (e.g. Mutual terms, standard SOC2 audit, healthy cash reserve)\n\n"
            "### 3. 💼 Core Obligations & Financial Commitments\n"
            "- Monetary obligations, fees, pricing, or revenue figures\n"
            "- Key deadlines, delivery milestones, or SLAs\n\n"
            "### 4. 🛡️ Regulatory, Privacy & Security Assessment\n"
            "- Compliance frameworks (SOC 2, ISO 27001, GDPR, SEC regulations)\n"
            "- Incident notification timelines & audit rights\n\n"
            "### 5. 🎯 Auditor Strategic Recommendations\n"
            "- 3 to 4 actionable recommendations for the executive leadership team prior to signing or investing.\n\n"
            "STRICT RULE: Ground every single claim in the provided document text and cite the Page numbers."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": f"Document Text for Audit:\n\n{combined_doc_context}\n\nPlease generate the formal Executive Due Diligence Report.",
            },
        ]

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=2500,
            temperature=0.2,
        )

        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Legacy backwards compatibility methods
    # ------------------------------------------------------------------

    def ask_a_question_from_pdf_stream(
        self,
        pdf_path: str,
        prompt: str = "tell me what is this pdf about",
        local_embedding=None,
    ):
        """
        Backwards-compatible streaming generator for legacy callers.
        """
        generator, _ = self.ask_question_stream(
            question=prompt,
            local_embedding=local_embedding,
            domain_mode="GENERAL_DUE_DILIGENCE",
            k=3,
        )
        for token in generator:
            yield token
