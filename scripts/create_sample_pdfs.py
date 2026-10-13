"""
Generates realistic enterprise business PDF documents for testing and demonstration:
1. SaaS Master Services Agreement (MSA) - Legal & Contracts Domain
2. Q1 Corporate Financial & Earnings Report - Financial Intelligence Domain
3. Corporate Cybersecurity & AI Governance Policy - Compliance Domain
"""

import os
from fpdf import FPDF


class ModernPDF(FPDF):
    def __init__(self, title_text: str, domain_text: str):
        super().__init__()
        self.doc_title = title_text
        self.doc_domain = domain_text
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 8, f"CONFIDENTIAL // {self.doc_domain.upper()} // {self.doc_title.upper()}", border=0, align="L")
        self.ln(6)
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.3)
        self.line(10, 16, 200, 16)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()} | DocuMind Enterprise Due Diligence Repository", align="C")


def create_legal_contract_pdf(output_path: str):
    pdf = ModernPDF("Master Services Agreement", "Legal & Contract Risk")
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "ENTERPRISE MASTER SERVICES AGREEMENT (MSA)", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 6, "Contract Reference: MSA-2026-CLOUD-9941 | Effective Date: January 15, 2026", ln=True, align="C")
    pdf.ln(8)

    # Parties
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, "1. PARTIES AND SCOPE OF AGREEMENT", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    text1 = (
        "This Enterprise Master Services Agreement ('Agreement') is entered into as of January 15, 2026 "
        "('Effective Date'), by and between CloudScale Technologies Inc., a Delaware corporation with its principal place "
        "of business at 500 Enterprise Way, Suite 400, San Francisco, CA ('Vendor' or 'Provider'), and "
        "OmniGlobal Enterprises LLC, a New York limited liability company with its principal office at 1250 Broadway, "
        "New York, NY ('Customer' or 'Client'). Vendor and Customer may collectively be referred to as the 'Parties'.\n\n"
        "Vendor provides an enterprise multi-tenant cloud data analytics and automated workflow orchestration platform. "
        "This Agreement establishes the general terms under which Customer may procure cloud subscription licenses, professional "
        "deployment services, and premium technical support from Vendor."
    )
    pdf.multi_cell(0, 5, text1)
    pdf.ln(4)

    # Section 2: Fees & Payment
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, "2. SUBSCRIPTION FEES, INVOICING, AND PAYMENT TERMS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text2 = (
        "2.1 Annual Subscription Fee: Customer shall pay Vendor an annual recurring subscription fee of $450,000 USD, "
        "payable net thirty (30) days from the date of invoice. All payment obligations are non-cancelable and fees paid are non-refundable.\n"
        "2.2 Overdue Payments: Any undisputed amount not received within thirty (30) days of the due date shall accrue interest at the "
        "rate of 1.5% per month or the maximum rate permitted by law, whichever is lower.\n"
        "2.3 Price Escalation: Upon renewal of the Initial Subscription Term, Vendor reserves the right to increase annual subscription "
        "fees by up to five percent (5%) upon providing sixty (60) days advance written notice."
    )
    pdf.multi_cell(0, 5, text2)
    pdf.ln(4)

    # Section 3: Data Security & Privacy
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "3. DATA SECURITY, COMPLIANCE, AND PRIVACY COVENANTS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text3 = (
        "3.1 Security Controls: Vendor maintains administrative, technical, and physical safeguards designed to protect Customer Data, "
        "including achieving and maintaining SOC 2 Type II certification and ISO 27001 compliance throughout the term.\n"
        "3.2 Security Breach Notification SLA: In the event of a verified Security Breach affecting Customer Data, Vendor shall notify "
        "Customer via email within forty-eight (48) hours of actual verification and take immediate remediation measures at its sole cost.\n"
        "3.3 Zero Data Retention for AI Features: Where AI automated inference features are activated, Customer Data processed through "
        "cloud inference endpoints is processed strictly in-memory without persistent storage, logging, or model retraining."
    )
    pdf.multi_cell(0, 5, text3)

    # Page 2
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "4. INTELLECTUAL PROPERTY RIGHTS AND RESTRICTIONS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text4 = (
        "4.1 Customer Data Ownership: Customer retains all right, title, and interest in Customer Data, including all intellectual "
        "property rights. Customer grants Vendor a limited, non-exclusive license solely to host, transmit, and display Customer Data as "
        "necessary to deliver the Services.\n"
        "4.2 Vendor Proprietary Platform: Vendor retains all rights in the platform, documentation, underlying models, algorithms, and APIs. "
        "Customer shall not reverse engineer, decompile, or create derivative works based on Vendor software."
    )
    pdf.multi_cell(0, 5, text4)
    pdf.ln(4)

    # Section 5: Limitation of Liability (CRITICAL RISK CLAUSE)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "5. LIMITATION OF LIABILITY (AUDIT WARNING: CAP CLAUSE)", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text5 = (
        "5.1 Exclusion of Consequential Damages: NEITHER PARTY SHALL BE LIABLE UNDER THIS AGREEMENT FOR ANY INDIRECT, INCIDENTAL, "
        "CONSEQUENTIAL, SPECIAL, OR PUNITIVE DAMAGES, INCLUDING LOSS OF PROFITS, REVENUE, OR DATA, REGARDLESS OF THE THEORY OF LIABILITY.\n"
        "5.2 Aggregate Cap on Liability: EXCEPT FOR GROSS NEGLIGENCE, WILLFUL MISCONDUCT, OR INDEMNIFICATION OBLIGATIONS UNDER SECTION 6, "
        "EACH PARTY'S MAXIMUM AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT SHALL BE STRICTLY LIMITED TO THE TOTAL AMOUNT "
        "ACTUALLY PAID BY CUSTOMER TO VENDOR IN THE TWELVE (12) MONTHS PRECEDING THE INCIDENT GIVING RISE TO LIABILITY, NOT TO EXCEED $500,000 USD."
    )
    pdf.multi_cell(0, 5, text5)
    pdf.ln(4)

    # Section 6: Indemnification
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "6. INDEMNIFICATION OBLIGATIONS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text6 = (
        "6.1 Vendor IP Indemnity: Vendor shall defend, indemnify, and hold harmless Customer against any third-party claim alleging that "
        "Customer's authorized use of the Services infringes or misappropriates any valid United States patent, copyright, or trademark.\n"
        "6.2 Customer Indemnity: Customer shall defend and indemnify Vendor against any third-party claim arising from Customer Data "
        "violating applicable privacy laws or infringing third-party intellectual property rights."
    )
    pdf.multi_cell(0, 5, text6)
    pdf.ln(4)

    # Section 7: Term & Termination
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "7. TERM, TERMINATION FOR CAUSE, AND TERMINATION FOR CONVENIENCE", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text7 = (
        "7.1 Term: The Initial Term of this Agreement is three (3) years starting January 15, 2026, expiring January 14, 2029.\n"
        "7.2 Termination for Convenience: Customer may terminate this Agreement without cause at any time by providing at least ninety (90) "
        "days advance written notice to Vendor; provided that Customer shall pay an early termination fee equal to 50% of the remaining fees.\n"
        "7.3 Termination for Cause: Either Party may terminate immediately upon written notice if the other Party materially breaches any term "
        "and fails to cure such breach within thirty (30) days of receiving written notice.\n"
        "7.4 Governing Law: This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, "
        "without regard to conflict of laws principles. The Parties submit to the exclusive jurisdiction of the state and federal courts in New Castle County, Delaware."
    )
    pdf.multi_cell(0, 5, text7)

    pdf.output(output_path)
    print(f"Generated: {output_path}")


def create_financial_earnings_pdf(output_path: str):
    pdf = ModernPDF("Q1 2026 Earnings Release", "Financial Intelligence")
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "APEX FINANCIAL DYNAMICS CORP. (NYSE: APEX)", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 6, "FIRST QUARTER 2026 FINANCIAL RESULTS & INVESTOR BRIEFING", ln=True, align="C")
    pdf.cell(0, 5, "Filing Date: April 14, 2026 | Conference Call Transcript & Earnings Release", ln=True, align="C")
    pdf.ln(8)

    # Executive Overview
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, "1. EXECUTIVE SUMMARY & KEY FINANCIAL HIGHLIGHTS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    text1 = (
        "Apex Financial Dynamics Corp. ('Apex' or the 'Company') today announced record financial performance for the first "
        "quarter ended March 31, 2026. Consolidated total revenue for Q1 2026 grew 18.4% year-over-year to $612.4 million, "
        "compared to $517.2 million in Q1 2025, driven by accelerating enterprise adoption of Apex Cloud Intelligence.\n\n"
        "Key Highlights for Q1 2026:\n"
        "- Total Revenue: $612.4 million, beating analyst consensus expectations of $595.0 million by 2.9%.\n"
        "- Annual Recurring Revenue (ARR): Reached $1.84 billion, representing year-over-year growth of 24.2%.\n"
        "- GAAP Gross Margin: 74.2%, expanding 180 basis points from 72.4% in the prior year period.\n"
        "- Non-GAAP Operating Income: $165.3 million (27.0% operating margin), compared to $124.1 million (24.0%) in Q1 2025.\n"
        "- Diluted Earnings Per Share (EPS): GAAP diluted EPS was $1.12, and Non-GAAP diluted EPS reached $1.44 vs $1.15 in Q1 2025.\n"
        "- Operating Cash Flow: $198.6 million, up 31.0% YoY; Free Cash Flow was $174.2 million (28.4% FCF margin)."
    )
    pdf.multi_cell(0, 5, text1)
    pdf.ln(4)

    # Segment Breakdown
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "2. BUSINESS SEGMENT PERFORMANCE", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text2 = (
        "2.1 Enterprise Cloud Division: Generated $440.8 million (72.0% of total revenue), reflecting 26.5% YoY expansion. "
        "The Company added 142 new customers with ARR greater than $100,000, bringing total enterprise customers to 2,890.\n"
        "2.2 Financial Analytics & Risk Suite: Revenue increased 7.1% YoY to $128.5 million. Net revenue retention rate (NRR) "
        "stood at 116%, indicating robust customer expansion.\n"
        "2.3 Professional Deployment Services: Contributed $43.1 million in revenue with a balanced gross margin of 21.0%."
    )
    pdf.multi_cell(0, 5, text2)

    # Page 2
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "3. CAPITAL ALLOCATION, BALANCE SHEET, AND LIQUIDITY", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text3 = (
        "3.1 Cash and Equivalents: Apex concluded the quarter with $1.42 billion in cash, cash equivalents, and short-term marketable securities, "
        "providing exceptional balance sheet flexibility.\n"
        "3.2 Debt Obligations: Total outstanding senior convertible notes stood at $450.0 million maturing in November 2028, with no debt "
        "maturities occurring within the next 24 months.\n"
        "3.3 Share Repurchase Program: During Q1 2026, the Company repurchased 1.8 million shares of Class A common stock for a total "
        "expenditure of $75.0 million. The Board of Directors authorized an additional $250.0 million expansion to the share buyback program."
    )
    pdf.multi_cell(0, 5, text3)
    pdf.ln(4)

    # Section 4: Guidance
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "4. FULL YEAR 2026 OUTLOOK & FORWARD GUIDANCE", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text4 = (
        "Based on first-quarter strength and high recurring revenue visibility, Apex is raising full-year 2026 guidance:\n"
        "- FY 2026 Total Revenue: Expected in the range of $2.58 billion to $2.62 billion, representing 17% to 19% YoY growth.\n"
        "- Non-GAAP Operating Margin: Projected between 27.5% and 28.5%.\n"
        "- Non-GAAP Diluted EPS: Projected between $6.15 and $6.30 per share.\n"
        "- Capital Expenditures (CapEx): Projected at $110 million to $125 million, primarily supporting AI GPU cluster deployment."
    )
    pdf.multi_cell(0, 5, text4)
    pdf.ln(4)

    # Section 5: Risk Factors
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "5. KEY AUDIT RISK FACTORS & MACROECONOMIC HEADWINDS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text5 = (
        "5.1 Enterprise Procurement Scrutiny: Prolonged enterprise decision-making cycles and multi-level budget sign-offs for deals "
        "exceeding $1.0 million ARR may impact quarterly timing of large bookings.\n"
        "5.2 Foreign Exchange Exposure: Approximately 28% of quarterly revenue originated outside North America. Continued strengthening "
        "of the US Dollar could impose a 150 to 200 basis point headwind on reported international growth rates.\n"
        "5.3 AI Infrastructure Cost Inflation: Rising costs of specialized cloud computing clusters and high-bandwidth interconnects "
        "could compress cloud gross margins if usage outpaces contractual tier limits."
    )
    pdf.multi_cell(0, 5, text5)

    pdf.output(output_path)
    print(f"Generated: {output_path}")


def create_cybersecurity_policy_pdf(output_path: str):
    pdf = ModernPDF("Cybersecurity & AI Policy", "Corporate Compliance")
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "VANGUARD ENTERPRISE SECURITY & AI GOVERNANCE SOP", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 6, "Policy Document: POL-SEC-2026-v4.2 | Classification: RESTRICTED INTERNAL", ln=True, align="C")
    pdf.cell(0, 5, "Effective Date: February 1, 2026 | Mandatory Annual Compliance Audit", ln=True, align="C")
    pdf.ln(8)

    # Section 1: Objective & Scope
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, "1. PURPOSE, SCOPE, AND GOVERNING STANDARDS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    text1 = (
        "1.1 Purpose: This Standard Operating Procedure establishes corporate requirements for securing enterprise data, "
        "governing generative AI usage, and ensuring regulatory compliance across all corporate business units.\n"
        "1.2 Applicable Standards: Vanguard Systems Corp complies with ISO/IEC 27001:2022, SOC 2 Type II Trust Services Criteria, "
        "NIST Cybersecurity Framework 2.0, and GDPR / CCPA data privacy statutes.\n"
        "1.3 Applicability: This policy applies to all full-time employees, contractors, third-party vendors, and system administrators."
    )
    pdf.multi_cell(0, 5, text1)
    pdf.ln(4)

    # Section 2: Access Control
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "2. IDENTITY MANAGEMENT, ZERO TRUST & AUTHENTICATION", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text2 = (
        "2.1 Multi-Factor Authentication (MFA): Phishing-resistant FIDO2 hardware security keys or authorized mobile authenticator apps "
        "are mandatory for accessing all enterprise systems, email, and VPN tunnels. SMS-based 2FA is strictly prohibited.\n"
        "2.2 Least Privilege Principle: Access to production databases and customer data repositories is granted on a Just-In-Time (JIT) "
        "basis with automatic revocation after four (4) hours. Quarterly access reviews are mandatory for all Department Heads.\n"
        "2.3 Password Policy: System accounts require minimum 16-character passphrases, automated 90-day compromise scanning, and "
        "immediate invalidation upon employee separation."
    )
    pdf.multi_cell(0, 5, text2)
    pdf.ln(4)

    # Section 3: AI Governance
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "3. GENERATIVE AI AND LARGE LANGUAGE MODEL GOVERNANCE", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text3 = (
        "3.1 Prohibition of Public AI Tools: Employees are strictly prohibited from submitting proprietary source code, Customer PII, "
        "unreleased financial data, or legal drafts into public AI chat interfaces (e.g., consumer ChatGPT, Claude, Gemini).\n"
        "3.2 Approved Enterprise RAG Pipelines: Employees may exclusively utilize approved internal RAG platforms that enforce "
        "Zero Data Retention (ZDR) APIs, local vector embedding models, and encrypted TLS 1.3 data transmission.\n"
        "3.3 Model Output Verification: Any AI-generated code, legal analysis, or regulatory report must undergo mandatory human review "
        "by a certified domain professional prior to external release or production deployment."
    )
    pdf.multi_cell(0, 5, text3)

    # Page 2
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "4. INCIDENT RESPONSE, CLASSIFICATION, AND NOTIFICATION SLAS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text4 = (
        "4.1 Severity Classification:\n"
        "- P1 Critical: Confirmed unauthorized exfiltration of Customer PII, ransomware deployment, or complete production outage.\n"
        "- P2 High: Suspected unauthorized credential compromise or malicious malware detection on corporate endpoints.\n"
        "- P3 Medium: Isolated policy violation or unauthorized software installation.\n"
        "4.2 Response Timelines:\n"
        "- P1 Critical Incidents require Incident Response Team activation within fifteen (15) minutes of alert generation.\n"
        "- Customer and Regulatory Notification SLA: Executive notification to affected customers and supervisory authorities must occur "
        "within seventy-two (72) hours of confirmed breach, in accordance with GDPR Article 33.\n"
        "4.3 Post-Mortem Requirement: A formal root-cause analysis (RCA) and remediation plan must be submitted to the Chief Information "
        "Security Officer (CISO) within five (5) business days of incident resolution."
    )
    pdf.multi_cell(0, 5, text4)
    pdf.ln(4)

    # Section 5: Audit & Sanctions
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "5. AUDIT CADENCE, PENALTIES, AND DISCIPLINARY ACTIONS", ln=True)
    pdf.set_font("Helvetica", "", 10)
    text5 = (
        "5.1 Audit Cadence: Third-party independent SOC 2 Type II audit occurs annually in October. Bi-annual external penetration testing "
        "is conducted across all external-facing cloud IP ranges.\n"
        "5.2 Non-Compliance Sanctions: Failure to adhere to identity or AI governance protocols constitutes a security violation subject to "
        "mandatory retraining, formal written reprimand, or termination of employment for willful gross negligence."
    )
    pdf.multi_cell(0, 5, text5)

    pdf.output(output_path)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    os.makedirs("sample_documents", exist_ok=True)
    create_legal_contract_pdf("sample_documents/saas_master_services_agreement.pdf")
    create_financial_earnings_pdf("sample_documents/q1_corporate_earnings_report.pdf")
    create_cybersecurity_policy_pdf("sample_documents/corporate_cybersecurity_policy.pdf")
    print("All enterprise sample PDFs successfully generated!")
