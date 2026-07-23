"""
Export missing_value_cca_analysis.html  →  missing_value_cca_analysis.pdf
Uses the Microsoft Edge / Chrome DevTools Protocol (no external installs needed).
"""
import subprocess, os, sys, time, glob

HTML_FILE = os.path.abspath("missing_value_cca_analysis.html")
PDF_FILE  = os.path.abspath("missing_value_cca_analysis.pdf")

# ── Find a Chromium-based browser ─────────────────────────────────────────────
CANDIDATES = [
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

browser_path = next((p for p in CANDIDATES if os.path.exists(p)), None)

if browser_path is None:
    print("[ERROR] No Chrome or Edge browser found. Cannot export to PDF.")
    print("  Please open 'missing_value_cca_analysis.html' in your browser")
    print("  and use Ctrl+P → Save as PDF.")
    sys.exit(1)

print("Using browser:", browser_path)

# ── Headless print-to-PDF via --print-to-pdf flag ────────────────────────────
cmd = [
    browser_path,
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--run-all-compositor-stages-before-draw",
    "--print-to-pdf={}".format(PDF_FILE),
    "--no-pdf-header-footer",
    "file:///{}".format(HTML_FILE.replace("\\", "/")),
]

print("Generating PDF (this may take 10-30 seconds)...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

if os.path.exists(PDF_FILE) and os.path.getsize(PDF_FILE) > 1000:
    size_kb = os.path.getsize(PDF_FILE) / 1024
    print("\n[SUCCESS] PDF exported!")
    print("  File : {}".format(PDF_FILE))
    print("  Size : {:.1f} KB".format(size_kb))
else:
    print("[ERROR] PDF generation failed.")
    print("STDERR:", result.stderr[:500] if result.stderr else "(none)")
    print("\n  Fallback: Open the HTML file in your browser and press Ctrl+P → Save as PDF")
    print("  HTML : {}".format(HTML_FILE))
