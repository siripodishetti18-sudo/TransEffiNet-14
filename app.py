import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from model import load_model, run_inference, CLASSES, get_severity, THRESHOLD

MODEL = load_model("best_hybrid_model.pth")


def make_disease_chart(probs):
    detected = []
    for i, disease in enumerate(CLASSES):
        sev, icon = get_severity(probs[i])
        if sev != "NORMAL":
            detected.append((disease, probs[i], sev))

    if not detected:
        fig, ax = plt.subplots(figsize=(7, 2))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.text(0.5, 0.5, "No significant findings detected",
                ha="center", va="center", fontsize=13,
                color="#2d6a4f", fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        return fig

    detected = sorted(detected, key=lambda x: x[1], reverse=True)
    color_map = {
        "SEVERE":   "#e63946",
        "MODERATE": "#f4a261",
        "MILD":     "#e9c46a",
    }
    diseases = [d[0] for d in detected]
    probs_   = [d[1] for d in detected]
    colors   = [color_map[d[2]] for d in detected]
    labels   = [f"{d[2]}  {d[1]*100:.1f}%" for d in detected]

    fig_h = max(2.5, len(detected) * 0.7 + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    bars = ax.barh(diseases[::-1], probs_[::-1],
                   color=colors[::-1], height=0.55,
                   edgecolor="white", linewidth=1.5)
    ax.axvline(THRESHOLD, color="#adb5bd", linestyle="--",
               linewidth=1, label=f"Threshold ({THRESHOLD})")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Confidence Score", fontsize=10, color="#495057")
    ax.set_title("Detected Conditions", fontsize=13,
                 fontweight="bold", color="#212529", pad=10)
    ax.tick_params(colors="#495057", labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#dee2e6")
    for bar, label in zip(bars, labels[::-1]):
        ax.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9,
                color="#212529", fontweight="bold")
    plt.tight_layout()
    return fig


def build_summary_html(probs):
    detected = []
    for i, disease in enumerate(CLASSES):
        sev, icon = get_severity(probs[i])
        if sev != "NORMAL":
            detected.append((disease, probs[i], sev, icon))

    detected = sorted(detected, key=lambda x: x[1], reverse=True)

    if not detected:
        return """
        <div style="background:#d4edda; border:1px solid #c3e6cb;
                    border-radius:10px; padding:1.2rem; text-align:center;">
            <span style="font-size:1.5rem;">✅</span>
            <p style="color:#155724; font-weight:700; font-size:1.1rem; margin:0.3rem 0 0;">
                No significant findings detected
            </p>
            <p style="color:#155724; font-size:0.85rem; margin:0.2rem 0 0;">
                All disease probabilities below threshold
            </p>
        </div>"""

    color_map = {
        "SEVERE":   ("🔴", "#fff5f5", "#e63946", "#c1121f"),
        "MODERATE": ("🟠", "#fff8f0", "#f4a261", "#e07c24"),
        "MILD":     ("🟡", "#fffdf0", "#e9c46a", "#b5850a"),
    }

    cards = ""
    for disease, prob, sev, _ in detected:
        icon, bg, border, text = color_map[sev]
        cards += f"""
        <div style="background:{bg}; border:2px solid {border};
                    border-radius:10px; padding:0.9rem 1.2rem;
                    margin-bottom:0.6rem; display:flex;
                    justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:1.1rem;">{icon}</span>
                <span style="font-weight:700; font-size:1rem;
                             color:{text}; margin-left:0.4rem;">{disease}</span>
                <span style="display:block; font-size:0.78rem;
                             color:#6c757d; margin-top:0.2rem; margin-left:1.6rem;">{sev}</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:1.3rem; font-weight:800; color:{text};">
                    {prob*100:.1f}%
                </span>
                <span style="display:block; font-size:0.75rem; color:#6c757d;">confidence</span>
            </div>
        </div>"""

    overall_sev = detected[0][2]
    overall_msgs = {
        "SEVERE":   ("🔴 HIGH RISK", "#fff5f5", "#e63946", "#c1121f",
                     "Immediate radiologist review recommended."),
        "MODERATE": ("🟠 MODERATE RISK", "#fff8f0", "#f4a261", "#e07c24",
                     "Follow-up with a radiologist recommended."),
        "MILD":     ("🟡 LOW RISK", "#fffdf0", "#e9c46a", "#b5850a",
                     "Monitor symptoms and recheck if needed."),
    }
    label, bg, border, text, msg = overall_msgs[overall_sev]

    return f"""
    <div style="background:{bg}; border:2px solid {border};
                border-radius:10px; padding:0.9rem 1.2rem;
                margin-bottom:1rem; text-align:center;">
        <p style="font-size:1.1rem; font-weight:800; color:{text}; margin:0;">{label}</p>
        <p style="font-size:0.85rem; color:{text}; margin:0.2rem 0 0;">{msg}</p>
    </div>
    {cards}
    <div style="background:#f8f9fa; border-radius:8px; padding:0.7rem 1rem;
                margin-top:0.8rem; font-size:0.75rem; color:#6c757d; text-align:center;">
        AI research demo only. Not for clinical diagnosis.
        Always consult a licensed radiologist.
    </div>"""


def analyze(image):
    if image is None:
        return None, None, None, "<p style='color:#6c757d;'>Upload an image to begin.</p>"
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    original, overlay, top_disease, probs, _ = run_inference(pil_image, MODEL)
    chart   = make_disease_chart(probs)
    summary = build_summary_html(probs)
    return original, overlay, chart, summary


with gr.Blocks() as demo:

    gr.HTML("""
    <div style="text-align:center; padding:1.5rem 0 0.5rem;
                border-bottom:2px solid #e9ecef; margin-bottom:1rem;">
        <h1 style="font-size:1.9rem; font-weight:800; color:#1d3557; margin:0;">
            🫁 Chest X-Ray AI Analyzer
        </h1>
        <p style="color:#6c757d; font-size:0.92rem; margin:0.4rem 0 0;">
            EfficientNet-B4 + Transformer &nbsp;·&nbsp;
            NIH ChestX-ray14 &nbsp;·&nbsp;
            Grad-CAM Explainability &nbsp;·&nbsp;
            14 Lung Diseases
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            inp = gr.Image(
                label="Upload Chest X-Ray",
                type="pil",
                height=260
            )
            btn = gr.Button("🔍  Analyze X-Ray", size="lg", variant="primary")
            gr.HTML("""
            <div style="background:#f1faee; border-radius:8px; padding:0.9rem;
                        margin-top:0.8rem; font-size:0.82rem;
                        color:#457b9d; line-height:1.8;">
                <b>How it works:</b><br>
                1. Upload a frontal chest X-ray<br>
                2. Click <b>Analyze X-Ray</b><br>
                3. Get heatmap + detected conditions<br><br>
                <b>Detects 14 conditions including:</b><br>
                Pneumothorax &middot; Effusion &middot; Cardiomegaly
                &middot; Pneumonia &middot; Edema &middot; Atelectasis
                &middot; and more
            </div>
            """)

        with gr.Column(scale=2):
            with gr.Row():
                out_original = gr.Image(
                    label="Enhanced X-Ray",
                    height=260
                )
                out_gradcam = gr.Image(
                    label="Grad-CAM Heatmap",
                    height=260
                )

            out_summary = gr.HTML(
                value="<p style='color:#adb5bd; text-align:center; padding:1rem;'>"
                      "Results will appear here after analysis.</p>"
            )

            out_chart = gr.Plot(label="Confidence Scores")

    btn.click(
        fn=analyze,
        inputs=[inp],
        outputs=[out_original, out_gradcam, out_chart, out_summary]
    )

    gr.HTML("""
    <div style="text-align:center; padding:1rem 0 0.5rem;
                border-top:1px solid #e9ecef; margin-top:1rem;
                font-size:0.78rem; color:#adb5bd;">
        Model: EfficientNet-B4 + Transformer &nbsp;|&nbsp;
        Loss: Focal Loss (&gamma;=2) &nbsp;|&nbsp;
        Dataset: NIH ChestX-ray14 (112,120 images) &nbsp;|&nbsp;
        CEECT 2026 &mdash; Paper ID 1534
    </div>
    """)

demo.launch()
