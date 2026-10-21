"""QMT-PAT Gastritis Classifier - Streamlit version.

Same model, samples and design as the Gradio app: give it all endoscopy images of one
patient and QMT-PAT predicts the four labels jointly. Checkpoints are read from
checkpoints/ if present, otherwise downloaded from the Hugging Face Hub.
"""
import streamlit as st
from PIL import Image

from config import FOLDS
from inference import SAMPLES_DIR, TASK_TITLES, label_of, load_models, load_samples, predict

st.set_page_config(page_title="QMT-PAT Gastritis Classifier", page_icon="🔬", layout="wide")

TASK_ORDER = ["H_pylori_infection_status", "Presence_of_atrophy",
              "Distribution_of_atrophy", "Gastritis_type"]

PRIMARY = "#1E9A7C"
PRIMARY_DARK = "#147A63"
ACCENT = "#E8B84B"
BG_CARD = "#FFFFFF"
TEXT_MUTED = "#5B6B67"

CUSTOM_CSS = f"""
<style>
.block-container {{ padding-top: 4rem; max-width: 1300px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.hero {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
    padding: 2.2rem 2.5rem; border-radius: 18px; color: white; margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(15,110,92,0.25);
}}
.hero h1 {{ margin: 0; padding: 0; font-size: 2.1rem; font-weight: 700; color: white; }}
.hero p {{ margin: 0.5rem 0 0 0; opacity: 0.92; font-size: 1.02rem; color: white; }}
.badge-row {{ margin-top: 1rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }}
.badge {{
    background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.35);
    padding: 0.3rem 0.8rem; border-radius: 999px; font-size: 0.82rem; color: white;
}}
.about-panel, .task-card {{
    background: {BG_CARD}; border-radius: 16px; padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #E7ECEA;
}}
.about-panel {{ margin-bottom: 1.2rem; font-size: 0.95rem; color: #1F2A27; }}
.results-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
@media (max-width: 700px) {{ .results-grid {{ grid-template-columns: 1fr; }} }}
.task-title {{
    font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: {TEXT_MUTED}; font-weight: 600; margin-bottom: 0.3rem;
}}
.task-pred {{ font-size: 1.5rem; font-weight: 700; color: {PRIMARY_DARK}; margin-bottom: 0.6rem; }}
.conf-track {{
    background: #EAF1EF; border-radius: 999px; height: 10px; width: 100%;
    overflow: hidden; margin-bottom: 0.4rem;
}}
.conf-fill {{ background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%); height: 100%; border-radius: 999px; }}
.conf-label {{ font-size: 0.85rem; color: {TEXT_MUTED}; margin-bottom: 0.7rem; }}
.all-probs {{ font-size: 0.82rem; color: {TEXT_MUTED}; }}
.all-probs summary {{ cursor: pointer; color: {PRIMARY_DARK}; font-weight: 600; }}
.prob-row {{ display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem; }}
.prob-row .prob-name {{ width: 40%; flex-shrink: 0; }}
.prob-row .prob-track {{ flex-grow: 1; background: #EAF1EF; border-radius: 999px; height: 8px; overflow: hidden; }}
.prob-row .prob-fill {{ background: {PRIMARY}; height: 100%; border-radius: 999px; }}
.prob-row .prob-fill.anchor {{ background: #9FB8B1; }}
.prob-row .prob-pct {{ width: 3rem; text-align: right; flex-shrink: 0; }}
.results-caption {{ color: {TEXT_MUTED}; font-size: 0.9rem; margin-bottom: 0.8rem; }}
.truth {{ font-size: 0.88rem; margin: -0.2rem 0 0.7rem 0; }}
.truth.ok {{ color: {PRIMARY_DARK}; }}
.truth.bad {{ color: #B4432F; }}
.delta-note {{ font-size: 0.8rem; color: {TEXT_MUTED}; margin-top: 0.6rem; }}
.banner {{ border-radius: 12px; padding: 0.8rem 1.1rem; font-size: 0.9rem; margin-bottom: 1rem; color: #1F2A27; }}
.banner.ok {{ background: #E8F5F1; border: 1px solid #BFE2D7; }}
.banner.bad {{ background: #FFF4EC; border: 1px solid #F3D2BC; }}
.placeholder {{ color: {TEXT_MUTED}; }}
.disclaimer {{
    background: #FFF8E8; border: 1px solid #F0DDA6; border-radius: 12px;
    padding: 0.9rem 1.1rem; font-size: 0.85rem; color: #6B5518; margin-top: 1.5rem;
}}
</style>
"""

HERO_HTML = """
<div class="hero">
  <h1>QMT-PAT Gastritis Classifier</h1>
  <p>Upload all endoscopy images of one patient for a joint read across four clinical dimensions,
  powered by a hybrid quantum-classical model: Vision Transformer features, patient-level attention
  and a 16-qubit task-structured circuit.</p>
  <div class="badge-row">
    <span class="badge">H. pylori status</span>
    <span class="badge">Presence of atrophy</span>
    <span class="badge">Distribution of atrophy</span>
    <span class="badge">Gastritis type</span>
    <span class="badge">16-qubit circuit</span>
  </div>
</div>
"""

ABOUT_HTML = f"""
<div class="about-panel">
<p>Gastritis, inflammation of the stomach lining, is one of the clearest early warning
signs doctors look for when assessing a patient's risk of developing stomach cancer down
the line. Left unaddressed, especially when driven by <i>H. pylori</i> infection, it can
progress through a well-documented sequence of increasingly serious changes to the
stomach lining. Catching and correctly characterizing it early matters.</p>
<p>This app runs <b>QMT-PAT</b>, a hybrid quantum-classical model that reads <b>all</b> of a
patient's endoscopy images together and answers the four questions a gastroenterologist would ask:</p>
<ul>
<li>Is <i>H. pylori</i> infection present?</li>
<li>Has the stomach lining started to thin out (atrophy)?</li>
<li>If so, where in the stomach is that thinning concentrated?</li>
<li>Which broader type of gastritis does the pattern most resemble, autoimmune,
or driven by <i>H. pylori</i>?</li>
</ul>
<p><b>How it works.</b> A fine-tuned Vision Transformer (ViT-B/16) turns each image into a feature
vector; a patient-level attention model pools all images into one summary per question and makes a
first prediction; a simulated <b>16-qubit circuit</b> (4 qubits per question, entangled across questions)
then adds a small learned correction. In five-fold cross-validation on EGID it reached a mean
balanced accuracy of 0.791 (AUROC 0.920).</p>
<p style="color:{TEXT_MUTED}; font-size:0.85rem; margin-bottom:0;">Research prototype, not a diagnostic device.
Not clinically validated. Should never inform real patient care.</p>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer">
<b>Research prototype, not a diagnostic device.</b> This tool is built for educational and research
purposes on the EGID dataset and has not been clinically validated. Predictions should never be used
to inform real patient care or replace professional medical judgement.
</div>
"""


# ---------------------------------------------------------------- results HTML
def _prob_rows(probs, cls_class=""):
    return "".join(
        f'<div class="prob-row"><div class="prob-name">{cls}</div>'
        f'<div class="prob-track"><div class="prob-fill {cls_class}" style="width:{p*100:.1f}%"></div></div>'
        f'<div class="prob-pct">{p*100:.1f}%</div></div>'
        for cls, p in probs.items()
    )


def _task_card_html(task, r, truth=None):
    conf_pct = r["confidence"] * 100
    truth_html = ""
    if truth:
        ok = truth == r["predicted_raw"]
        mark, cls = ("✓", "ok") if ok else ("✗", "bad")
        truth_html = f'<div class="truth {cls}">{mark} Ground truth: <b>{label_of(task, truth)}</b></div>'
    return f"""
    <div class="task-card">
        <div class="task-title">{TASK_TITLES[task]}</div>
        <div class="task-pred">{r['predicted_class']}</div>
        <div class="conf-track"><div class="conf-fill" style="width:{conf_pct:.1f}%"></div></div>
        <div class="conf-label">{conf_pct:.1f}% confidence</div>
        {truth_html}
        <details class="all-probs">
            <summary>See all class probabilities</summary>
            {_prob_rows(r["all_probs"])}
        </details>
    </div>
    """


def _results_html(results, hierarchy, caption, truth=None):
    if hierarchy["consistent"]:
        banner = (f'<div class="banner ok">✓ Consistent with the label hierarchy: '
                  f'<b>{hierarchy["distribution"]}</b> atrophy → <b>{hierarchy["expected_atrophy"]}</b>, '
                  f'<b>{hierarchy["expected_type"]}</b>.</div>')
    else:
        banner = (f'<div class="banner bad">⚠ The four predictions disagree with the label hierarchy: '
                  f'<b>{hierarchy["distribution"]}</b> atrophy implies <b>{hierarchy["expected_atrophy"]}</b> and '
                  f'<b>{hierarchy["expected_type"]}</b>.</div>')
    cards = "".join(_task_card_html(t, results[t], truth[t] if truth else None) for t in TASK_ORDER)
    return f'<div class="results-caption">{caption}</div>{banner}<div class="results-grid">{cards}</div>'


# ---------------------------------------------------------------- model + samples
@st.cache_resource(show_spinner="Loading QMT-PAT (ViT-B/16 + 16-qubit circuit) ...")
def get_models():
    return load_models()


@st.cache_data
def get_samples():
    return load_samples()


def sample_files(key):
    return sorted((SAMPLES_DIR / key).glob("*.jpg"))


def run(images, truth=None, source=""):
    with st.spinner("Extracting ViT features and running the quantum circuit ..."):
        results, hierarchy, n_models, n_images = predict(images)
    if n_models > 1:
        caption = f"{source}{n_images} image(s), {n_models} QMT-PAT models (folds {', '.join(map(str, FOLDS))}) averaged."
    else:
        caption = f"{source}{n_images} image(s) read together as one patient by QMT-PAT (fold {FOLDS[0]} model)."
    if n_images > 108:
        caption += " 108 evenly spaced images used."
    st.session_state["results_html"] = _results_html(results, hierarchy, caption, truth)


# ---------------------------------------------------------------- page
st.html(CUSTOM_CSS)
st.html(HERO_HTML)
st.html(ABOUT_HTML)

try:
    get_models()
except Exception as e:  # missing checkpoints / no network
    st.error(
        f"Could not load the model checkpoints: {e}\n\n"
        "Put `vit_b16_fold0_best.pt` and `v9_quantum_fold0_best.pt` in `checkpoints/`, "
        "or upload them to the Hugging Face repo set in `config.py`."
    )
    st.stop()

samples = get_samples()
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown("#### Input image(s)")
    st.markdown(
        "Give the model **all** images from one patient's exam. QMT-PAT reads them together "
        "and makes one patient-level prediction, exactly as it was validated."
    )
    tab_sample, tab_upload = st.tabs(["Try a sample patient", "Upload your own"])

    with tab_sample:
        keys = list(samples)
        if not keys:
            st.info("No sample patients found.")
        else:
            key = st.selectbox(
                "Sample patient",
                keys,
                format_func=lambda k: f"Patient {samples[k]['patient_id']} — "
                f"{label_of('Distribution_of_atrophy', samples[k]['labels']['Distribution_of_atrophy'])}"
                f" ({samples[k]['n_images']} images)",
            )
            files = sample_files(key)
            with st.container(height=230, border=True):
                cols = st.columns(3)
                for i, f in enumerate(files):
                    cols[i % 3].image(str(f), width="stretch")
            if st.button("Analyze sample patient", type="primary", width="stretch"):
                meta = samples[key]
                run([Image.open(f) for f in files], source=f"Sample patient {meta['patient_id']}: ")
            st.caption(
                "Sample patients are held-out test patients of fold 0, never seen in training "
                "by this model. Images: EGID dataset, CC BY-NC-ND 4.0."
            )

    with tab_upload:
        uploads = st.file_uploader(
            "Upload endoscopy image(s)",
            type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
        )
        if st.button("Analyze uploaded image(s)", type="primary", disabled=not uploads, width="stretch"):
            uploads = sorted(uploads, key=lambda f: f.name)
            run([Image.open(f) for f in uploads])

with right:
    st.markdown("#### Prediction results")
    st.html(st.session_state.get(
        "results_html",
        '<p class="placeholder">Pick a sample patient or upload image(s), then click Analyze.</p>',
    ))

st.html(DISCLAIMER_HTML)
