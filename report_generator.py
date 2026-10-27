"""
report_generator.py - Generates printable & downloadable Agronomy Diagnostic Reports
Formats results into an academic/laboratory diagnostic certificate with Grad-CAM visuals.
"""

import io
import base64
from datetime import datetime
from PIL import Image


def pil_to_base64(pil_image, format="JPEG", quality=85):
    """Converts PIL Image to Base64 data URL."""
    buffered = io.BytesIO()
    if format.upper() == "JPEG":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format=format, quality=quality)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def generate_diagnostic_html_report(
    sample_id,
    image_filename,
    predicted_class,
    confidence,
    severity_level,
    infection_pct,
    original_pil,
    heatmap_pil,
    overlay_pil,
    scores_list,
    disease_info,
    model_name="Improved Deep CNN (Custom Architecture)",
):
    """
    Renders a standalone, printable HTML diagnostic certificate for the crop scan.
    """
    orig_b64 = pil_to_base64(original_pil)
    heat_b64 = pil_to_base64(heatmap_pil)
    over_b64 = pil_to_base64(overlay_pil)

    now_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    # Format probability table rows
    prob_rows = ""
    for item in scores_list:
        bar_width = min(100.0, max(0.0, item["score"] * 100))
        prob_rows += f"""
        <tr>
            <td style="padding: 8px 12px; font-weight: 600; color: #1e3a29;">{item['class']}</td>
            <td style="padding: 8px 12px; text-align: right; font-weight: 700; color: #174c35;">{item['score'] * 100:.2f}%</td>
            <td style="padding: 8px 12px; width: 40%;">
                <div style="background: #e2ece3; border-radius: 6px; height: 12px; overflow: hidden;">
                    <div style="background: #24734c; width: {bar_width}%; height: 100%;"></div>
                </div>
            </td>
        </tr>
        """

    # Format symptoms list
    symptoms_html = "".join(f"<li style='margin-bottom: 6px;'>{s}</li>" for s in disease_info.get("symptoms", []))
    cultural_html = "".join(f"<li style='margin-bottom: 6px;'>{c}</li>" for c in disease_info.get("cultural_control", []))
    organic_html = "".join(f"<li style='margin-bottom: 6px;'>{o}</li>" for o in disease_info.get("organic_remedies", []))
    chemical_html = "".join(f"<li style='margin-bottom: 6px;'>{ch}</li>" for ch in disease_info.get("chemical_control", []))

    severity_badge_color = "#24734c" if predicted_class == "Healthy" else ("#c2410c" if infection_pct > 35 else "#ca8a04")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Diagnostic Report - {sample_id} - WheatCare AI</title>
<style>
    @page {{
        size: A4;
        margin: 1.5cm;
    }}
    body {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
        color: #1a2b21;
        background: #ffffff;
        margin: 0;
        padding: 24px;
        line-height: 1.5;
    }}
    .certificate {{
        max-width: 900px;
        margin: 0 auto;
        border: 2px solid #24734c;
        border-radius: 14px;
        padding: 32px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #d8e5d9;
        padding-bottom: 18px;
        margin-bottom: 24px;
    }}
    .logo-title h1 {{
        margin: 0;
        color: #174c35;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }}
    .logo-title p {{
        margin: 4px 0 0 0;
        color: #4b6855;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .meta-box {{
        text-align: right;
        font-size: 13px;
        color: #4b6855;
    }}
    .result-banner {{
        background: #f1f7f2;
        border-left: 6px solid #24734c;
        padding: 18px 24px;
        border-radius: 8px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .result-title {{
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        color: #4b6855;
        letter-spacing: 1px;
    }}
    .result-value {{
        font-size: 28px;
        font-weight: 800;
        color: #123d2c;
        margin: 4px 0;
    }}
    .result-pathogen {{
        font-size: 14px;
        font-style: italic;
        color: #2b543c;
    }}
    .badges {{
        text-align: right;
    }}
    .badge {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 700;
        color: #ffffff;
        background: #24734c;
        margin-bottom: 6px;
    }}
    .gallery {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }}
    .gallery-card {{
        background: #f9fbf9;
        border: 1px solid #dce8dd;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }}
    .gallery-card img {{
        width: 100%;
        height: 200px;
        object-fit: contain;
        border-radius: 6px;
        background: #ffffff;
    }}
    .gallery-card p {{
        margin: 8px 0 0 0;
        font-size: 12px;
        font-weight: 700;
        color: #3b5c47;
    }}
    .section-title {{
        color: #174c35;
        font-size: 17px;
        font-weight: 750;
        border-bottom: 1px solid #dce8dd;
        padding-bottom: 6px;
        margin: 22px 0 12px 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    th {{
        background: #eef5ef;
        padding: 8px 12px;
        text-align: left;
        color: #174c35;
        font-weight: 700;
    }}
    .prescription-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        font-size: 13px;
    }}
    .prescription-block {{
        background: #fcfdfc;
        border: 1px solid #e2ebe3;
        border-radius: 8px;
        padding: 14px 18px;
    }}
    .prescription-block h4 {{
        margin: 0 0 8px 0;
        color: #1b5338;
        font-size: 14px;
        font-weight: 700;
    }}
    .prescription-block ul {{
        margin: 0;
        padding-left: 18px;
    }}
    .footer-note {{
        margin-top: 30px;
        padding-top: 16px;
        border-top: 1px dashed #c8dacb;
        font-size: 11px;
        color: #6a8372;
        text-align: center;
    }}
    @media print {{
        body {{
            padding: 0;
        }}
        .certificate {{
            border: none;
            box-shadow: none;
            padding: 0;
        }}
        .no-print {{
            display: none !important;
        }}
    }}
</style>
</head>
<body>

<div class="certificate">
    <div class="header">
        <div class="logo-title">
            <h1>🌾 WheatCare AI Diagnostic Report</h1>
            <p>Final Year Engineering Research Project · Computer Vision & Deep Learning</p>
        </div>
        <div class="meta-box">
            <div><strong>Report ID:</strong> {sample_id}</div>
            <div><strong>Scan Timestamp:</strong> {now_str}</div>
            <div><strong>Input File:</strong> {image_filename}</div>
            <div><strong>Model Architecture:</strong> {model_name}</div>
        </div>
    </div>

    <div class="result-banner">
        <div>
            <div class="result-title">Primary Diagnostic Classification</div>
            <div class="result-value">{predicted_class}</div>
            <div class="result-pathogen">Pathogen: {disease_info.get('scientific_name', 'Triticum aestivum')} ({disease_info.get('category', 'Cereal Crop')})</div>
        </div>
        <div class="badges">
            <div class="badge">Confidence: {confidence * 100:.2f}%</div><br>
            <div class="badge" style="background: {severity_badge_color};">Severity: {severity_level} ({infection_pct:.1f}% Salient Area)</div>
        </div>
    </div>

    <div class="section-title">🔍 Explainable AI (Grad-CAM) Visual Evidence</div>
    <div class="gallery">
        <div class="gallery-card">
            <img src="{orig_b64}" alt="Original Leaf">
            <p>01 · Input Wheat Leaf Image</p>
        </div>
        <div class="gallery-card">
            <img src="{heat_b64}" alt="Grad-CAM Heatmap">
            <p>02 · Grad-CAM Activation Heatmap</p>
        </div>
        <div class="gallery-card">
            <img src="{over_b64}" alt="Superimposed Attention Overlay">
            <p>03 · Superimposed Diagnostic Overlay</p>
        </div>
    </div>

    <div class="section-title">📊 Multi-Class Probability Distribution</div>
    <table>
        <thead>
            <tr>
                <th>Class Label</th>
                <th style="text-align: right;">Softmax Score</th>
                <th>Confidence Distribution</th>
            </tr>
        </thead>
        <tbody>
            {prob_rows}
        </tbody>
    </table>

    <div class="section-title">🌿 Agronomic Clinical Action Plan & Prescription</div>
    <div class="prescription-grid">
        <div class="prescription-block">
            <h4>🔬 Typical Symptoms & Diagnostic Identifiers</h4>
            <ul>{symptoms_html}</ul>
        </div>
        <div class="prescription-block">
            <h4>🚜 Cultural & Agronomic Management</h4>
            <ul>{cultural_html}</ul>
        </div>
        <div class="prescription-block">
            <h4>🌱 Organic & Biological Remedies</h4>
            <ul>{organic_html}</ul>
        </div>
        <div class="prescription-block">
            <h4>🧪 Recommended Chemical Fungicides</h4>
            <ul>{chemical_html}</ul>
        </div>
    </div>

    <div class="footer-note">
        <strong>Academic Disclaimer:</strong> This report is generated by an academic Deep Learning CNN model (9-layer custom convolutional network trained on 4,800+ wheat leaf samples). While achieving 89.14% validation accuracy, predictions should be verified through in-field plant pathology before applying chemical interventions.
    </div>
</div>

<div class="no-print" style="text-align: center; margin-top: 24px;">
    <button onclick="window.print()" style="background: #24734c; color: #ffffff; border: none; padding: 12px 28px; border-radius: 8px; font-size: 15px; font-weight: 700; cursor: pointer; box-shadow: 0 4px 12px rgba(36,115,76,0.3);">
        🖨️ Print or Save as PDF
    </button>
</div>

</body>
</html>
"""
    return html
