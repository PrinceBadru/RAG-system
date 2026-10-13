import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By

html_content = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    body {
      background-color: #0b0f19;
      display: flex;
      justify-content: center;
      align-items: center;
      margin: 0;
      padding: 40px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    #diagram {
      background: #111827;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 16px;
      padding: 30px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
  </style>
</head>
<body>
  <div id="diagram">
    <pre class="mermaid">
    flowchart TD
        classDef ui fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#e0e7ff;
        classDef parser fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f0f9ff;
        classDef vector fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5;
        classDef llm fill:#4c1d95,stroke:#a855f7,stroke-width:2px,color:#faf5ff;
        classDef groq fill:#831843,stroke:#f43f5e,stroke-width:2px,color:#fff1f2;

        UI["🛡️ DocuMind AI SaaS UI<br/>(Streamlit Executive Dashboard)"]:::ui
        PR["📑 Page-Aware PDF Reader<br/>(Sliding-Window Chunker & Sanitizer)"]:::parser
        LE1["🧠 Local Embeddings (MiniLM-L6-v2)<br/>(384-dim L2 Normalised)"]:::vector
        VI[("⚡ Pure-Python Vector Index<br/>(In-Memory Cosine Similarity Store)")]:::vector
        LE2["🔍 Query Embedder<br/>(Local MiniLM Vectorizer)"]:::vector
        LC["📋 Enterprise LLM Client<br/>(Domain System Prompts & Exhibits)"]:::llm
        GROQ(["⚡ Groq Cloud Inference API<br/>(Qwen 3.8-27B · Zero Data Retention)"]):::groq

        UI -->|"1. PDF Upload / Enterprise Sample"| PR
        PR -->|"2. Pages & Overlapping Chunks"| LE1
        LE1 -->|"3. Batch Document Vectors"| VI

        UI -->|"4. Audit Inquiry / Quick Action"| LE2
        LE2 -->|"5. Search Vector"| VI
        VI -->|"6. Top-K Cited Exhibits (Page, Score)"| LC
        LC -->|"7. Grounded RAG Prompt"| GROQ
        GROQ -->|"8. High-Speed Streamed Tokens"| UI
    </pre>
  </div>
  <script>
    mermaid.initialize({
      startOnLoad: true,
      theme: 'dark',
      themeVariables: {
        darkMode: true,
        background: '#111827',
        primaryColor: '#6366f1',
        primaryTextColor: '#f8fafc',
        lineColor: '#64748b'
      }
    });
  </script>
</body>
</html>
"""

temp_html_path = os.path.abspath("scripts/diagram.html")
with open(temp_html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

c_opts = ChromeOptions()
c_opts.add_argument("--headless=new")
c_opts.add_argument("--window-size=1000,1400")
c_opts.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=c_opts)

try:
    file_url = f"file:///{temp_html_path.replace(os.sep, '/')}"
    print(f"Loading {file_url}...")
    driver.get(file_url)
    time.sleep(4)

    elem = driver.find_element(By.ID, "diagram")
    out_path = os.path.abspath("application_screenshots/mermaid_diagram.png")
    elem.screenshot(out_path)
    print(f"Saved updated mermaid diagram: {out_path} ({os.path.getsize(out_path)} bytes)")

finally:
    driver.quit()
    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)
