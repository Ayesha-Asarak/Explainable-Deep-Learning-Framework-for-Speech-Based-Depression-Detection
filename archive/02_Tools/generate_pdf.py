#!/usr/bin/env python3
"""Convert INTERVIEW_GUIDE.md to PDF."""

import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent
DEFAULT_MD = ROOT / "INTERVIEW_GUIDE.md"
DEFAULT_PDF = ROOT / "INTERVIEW_GUIDE.pdf"
DEFAULT_HTML = ROOT / "INTERVIEW_GUIDE.html"


def md_to_html(md_text: str, title: str = "Document") -> str:
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
    )
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
@page {{
    size: A4;
    margin: 1.8cm 1.5cm;
}}
body {{
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #1a1a1a;
}}
h1 {{
    font-size: 20pt;
    color: #1a365d;
    border-bottom: 2px solid #2b6cb0;
    padding-bottom: 6px;
    margin-top: 24px;
    page-break-before: always;
}}
h1:first-of-type {{ page-break-before: avoid; }}
h2 {{
    font-size: 14pt;
    color: #2c5282;
    margin-top: 18px;
    border-bottom: 1px solid #bee3f8;
    padding-bottom: 4px;
}}
h3 {{ font-size: 11pt; color: #2d3748; margin-top: 14px; }}
h4 {{ font-size: 10pt; color: #4a5568; }}
p {{ margin: 6px 0; }}
ul, ol {{ margin: 6px 0 6px 20px; }}
li {{ margin: 3px 0; }}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 9pt;
}}
th, td {{
    border: 1px solid #cbd5e0;
    padding: 5px 8px;
    text-align: left;
}}
th {{ background: #ebf8ff; font-weight: bold; }}
tr:nth-child(even) {{ background: #f7fafc; }}
code {{
    background: #edf2f7;
    padding: 1px 4px;
    font-family: Courier, monospace;
    font-size: 8.5pt;
}}
pre {{
    background: #2d3748;
    color: #e2e8f0;
    padding: 10px;
    font-size: 8pt;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    border-radius: 4px;
}}
blockquote {{
    border-left: 4px solid #4299e1;
    margin: 10px 0;
    padding: 8px 12px;
    background: #ebf8ff;
    font-style: italic;
}}
hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 16px 0; }}
strong {{ color: #1a365d; }}
a {{ color: #2b6cb0; text-decoration: none; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def html_to_pdf(html: str, pdf_path: Path) -> None:
    with open(pdf_path, "wb") as pdf_file:
        status = pisa.CreatePDF(html, dest=pdf_file, encoding="utf-8")
    if status.err:
        raise RuntimeError(f"PDF creation failed with {status.err} errors")


def convert(md_path: Path, pdf_path: Path, html_path: Path, title: str) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    html = md_to_html(md_text, title=title)
    html_path.write_text(html, encoding="utf-8")
    html_to_pdf(html, pdf_path)
    print(f"Created: {pdf_path}")
    print(f"Created: {html_path}")
    print(f"Size: {pdf_path.stat().st_size / 1024:.1f} KB")


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "supervisor":
        convert(
            ROOT / "SUPERVISOR_PROJECT_REPORT.md",
            ROOT / "SUPERVISOR_PROJECT_REPORT.pdf",
            ROOT / "SUPERVISOR_PROJECT_REPORT.html",
            "Supervisor Project Report - Speech Depression Detection",
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        convert(DEFAULT_MD, DEFAULT_PDF, DEFAULT_HTML, "Interview Guide")
        convert(
            ROOT / "SUPERVISOR_PROJECT_REPORT.md",
            ROOT / "SUPERVISOR_PROJECT_REPORT.pdf",
            ROOT / "SUPERVISOR_PROJECT_REPORT.html",
            "Supervisor Project Report",
        )
    else:
        convert(DEFAULT_MD, DEFAULT_PDF, DEFAULT_HTML, "Interview Guide")


if __name__ == "__main__":
    main()
