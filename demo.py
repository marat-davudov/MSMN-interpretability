"""
Streamlit demo for MSMN model interpretability.
Run with:  streamlit run demo.py
"""

import os
import re
import sys
import json
import torch
import numpy as np
import streamlit as st
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from constant import MSMN_MODEL_PATH, EMBEDDING_MODEL_PATH

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SAMPLE_DATA_DIR = "./sample_data"
MIMIC_3_DIR = os.path.join(SAMPLE_DATA_DIR, "mimic3")
EMBEDDING_DIR = "./embedding"

SYNTHETIC_NOTE = """\
Admission Date:  [**2023-03-15**]              Discharge Date:   [**2023-03-22**]

Service: MEDICINE

Chief Complaint:
Shortness of breath, chest pain

History of Present Illness:
72 year-old male with history of congestive heart failure, atrial fibrillation,
type 2 diabetes mellitus, hypertension, chronic kidney disease stage III, and
chronic obstructive pulmonary disease presenting with acute worsening dyspnea
and substernal chest pain for the past 2 days. Patient reports orthopnea,
paroxysmal nocturnal dyspnea, and bilateral lower extremity edema. He also
notes decreased urine output over the past week. His medications include
metformin, lisinopril, metoprolol, warfarin, and furosemide. He ran out of
furosemide one week ago.

On presentation, vitals: T 98.6F, HR 112 irregular, BP 165/95, RR 28,
SpO2 88% on room air. Physical exam notable for elevated JVP, bibasilar
crackles, 3+ pitting edema bilaterally, and irregularly irregular rhythm.

Pertinent Results:
BNP 1850 pg/mL, troponin 0.08 (stable), creatinine 2.1 (baseline 1.4),
potassium 5.2, HbA1c 8.9%. CXR with bilateral pleural effusions and
pulmonary vascular congestion. ECG with atrial fibrillation with rapid
ventricular response.

Hospital Course:
Patient was admitted to the medicine service for acute decompensated heart
failure with volume overload in the setting of medication non-compliance.
He was started on IV furosemide with good diuretic response. Metoprolol
was uptitrated for rate control of atrial fibrillation. Creatinine improved
to 1.6 with diuresis. Diabetes was managed with insulin sliding scale.
Warfarin was continued for anticoagulation.

Discharge Diagnoses:
Acute on chronic systolic congestive heart failure
Atrial fibrillation with rapid ventricular response
Type 2 diabetes mellitus, uncontrolled
Hypertensive heart disease
Acute kidney injury
Chronic kidney disease stage III
Chronic obstructive pulmonary disease

Discharge Medications:
Furosemide 80 mg PO daily
Metoprolol succinate 100 mg PO daily
Lisinopril 20 mg PO daily
Warfarin 5 mg PO daily
Insulin glargine 20 units subcutaneous at bedtime
Metformin 500 mg PO twice daily
"""

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Vocabulary helpers (mirrors data_util.load_vocab for word2vec .model files)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_vocab(vocab_path: str):
    """Build word2id / id2word from a gensim word2vec .model file.

    The companion .wv.vectors.npy file may be absent (e.g. only the main
    .model pickle was downloaded).  We only need the vocabulary ordering,
    not the actual float vectors (the checkpoint already has those baked in),
    so we temporarily patch np.load to return a dummy zero array for any
    missing companion file.
    """
    import gensim
    import numpy as np

    _orig_np_load = np.load

    def _safe_np_load(fname, mmap_mode=None, allow_pickle=False,
                      fix_imports=True, encoding="ASCII"):
        try:
            return _orig_np_load(fname, mmap_mode=mmap_mode,
                                 allow_pickle=allow_pickle,
                                 fix_imports=fix_imports, encoding=encoding)
        except (FileNotFoundError, IOError):
            # Return a placeholder zero array.  Shape (200000, 100) is large
            # enough for any MIMIC word2vec vocab; gensim stores the actual
            # vocabulary dict in the pickle, so key_to_index will be correct.
            return np.zeros((200000, 100), dtype=np.float32)

    np.load = _safe_np_load
    try:
        if vocab_path.endswith(".model"):
            model = gensim.models.Word2Vec.load(vocab_path)
        elif vocab_path.endswith(".bin"):
            model = gensim.models.KeyedVectors.load_word2vec_format(
                vocab_path, binary=True)
        else:
            raise ValueError(f"Unsupported vocab format: {vocab_path}")
    finally:
        np.load = _orig_np_load

    # gensim 4.x uses key_to_index; gensim 3.x uses vocab
    try:
        words = list(model.wv.key_to_index.keys())
    except AttributeError:
        words = list(model.wv.vocab.keys())

    # Trim to words that appear in the training corpus (same as data_util)
    word_count_path = os.path.join(EMBEDDING_DIR, "word_count_dict.json")
    if os.path.exists(word_count_path):
        import ujson
        with open(word_count_path, "r") as f:
            word_count_dict = ujson.load(f)
        words = [w for w in words if w in word_count_dict]

    for special in ["**UNK**", "**PAD**", "**MASK**"]:
        if special not in words:
            words.append(special)

    word2id = {w: i for i, w in enumerate(words)}
    id2word = {i: w for i, w in enumerate(words)}
    del model
    return word2id, id2word


# ---------------------------------------------------------------------------
# ICD code descriptions
# ---------------------------------------------------------------------------


@st.cache_resource
def load_code_descriptions():
    desc_dict = defaultdict(str)
    diag_path = os.path.join(SAMPLE_DATA_DIR, "D_ICD_DIAGNOSES.csv")
    proc_path = os.path.join(SAMPLE_DATA_DIR, "D_ICD_PROCEDURES.csv")
    icd9_path = os.path.join(SAMPLE_DATA_DIR, "ICD9_descriptions")

    import csv

    def reformat(code, is_diag):
        code = "".join(code.split("."))
        if is_diag:
            if code.startswith("E"):
                if len(code) > 4:
                    code = code[:4] + "." + code[4:]
            else:
                if len(code) > 3:
                    code = code[:3] + "." + code[3:]
        else:
            if len(code) > 2:
                code = code[:2] + "." + code[2:]
        return code

    if os.path.exists(diag_path):
        with open(diag_path, "r") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                desc_dict[reformat(row[1], True)] = row[-1]
    if os.path.exists(proc_path):
        with open(proc_path, "r") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                code = row[1]
                if code not in desc_dict:
                    desc_dict[reformat(code, False)] = row[-1]
    if os.path.exists(icd9_path):
        with open(icd9_path, "r") as f:
            for line in f:
                parts = line.rstrip().split()
                code = parts[0]
                if code not in desc_dict:
                    desc_dict[code] = " ".join(parts[1:])
    return desc_dict


# ---------------------------------------------------------------------------
# Build ind2c for mimic3-50 from the sample train JSON
# ---------------------------------------------------------------------------


@st.cache_resource
def load_ind2c():
    """Reproduce the sorted code→index mapping used during training."""
    import csv

    train_csv = os.path.join(MIMIC_3_DIR, "train_50.csv")
    train_json = os.path.join(MIMIC_3_DIR, "mimic3-50_train.json")

    codes = set()
    if os.path.exists(train_csv):
        with open(train_csv, "r") as f:
            lr = csv.reader(f)
            next(lr)
            for row in lr:
                for code in row[3].split(";"):
                    codes.add(code)
        for split in ["dev", "test"]:
            p = train_csv.replace("train", split)
            if os.path.exists(p):
                with open(p, "r") as f:
                    lr = csv.reader(f)
                    next(lr)
                    for row in lr:
                        for code in row[3].split(";"):
                            codes.add(code)
    elif os.path.exists(train_json):
        # Fallback: gather codes from JSON files
        import ujson
        for split in ["train", "dev", "test"]:
            jp = os.path.join(MIMIC_3_DIR, f"mimic3-50_{split}.json")
            if os.path.exists(jp):
                with open(jp, "r") as f:
                    data = ujson.load(f)
                for item in data:
                    for code in str(item["LABELS"]).split(";"):
                        codes.add(code)

    codes = set(c for c in codes if c != "")
    ind2c = {i: c for i, c in enumerate(sorted(codes))}
    return ind2c


# ---------------------------------------------------------------------------
# Tokeniser (mirrors MimicFullDataset)
# ---------------------------------------------------------------------------

_tokenizer = RegexpTokenizer(r"\w+")


def split_text(text):
    sp = re.sub(r"\n\n+|  +", "\t", text.strip()).replace(
        "\n", " ").replace("!", "\t").replace("?", "\t").replace(".", "\t")
    return [s.strip() for s in sp.split("\t") if s.strip()]


def tokenize(text):
    texts = split_text(text)
    all_text = []
    for note in texts:
        now_text = [w.lower() for w in _tokenizer.tokenize(note) if not w.isnumeric()]
        if now_text:
            all_text.extend(now_text)
    return all_text


def text_to_input(text, word2id, truncate_length=4000):
    tokens = tokenize(text)
    pad_id = word2id["**PAD**"]
    unk_id = word2id["**UNK**"]
    input_ids = [word2id.get(w, unk_id) for w in tokens]
    mask = [1] * len(input_ids)

    # Truncate / pad
    if len(input_ids) > truncate_length:
        input_ids = input_ids[:truncate_length]
        mask = mask[:truncate_length]
        tokens = tokens[:truncate_length]
    else:
        pad_len = truncate_length - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        mask = mask + [0] * pad_len

    return tokens, input_ids, mask


# ---------------------------------------------------------------------------
# Prepare label features (ICD code synonym descriptions for the decoder)
# ---------------------------------------------------------------------------


def prepare_label_features(word2id, ind2c, device, truncate_length=30, term_count=8, sort_method="random"):
    """Replicate MimicFullDataset.prepare_label_feature for mimic3-50."""
    desc_dict = load_code_descriptions()
    import ujson

    desc_list = []
    for i in sorted(ind2c.keys()):
        code = ind2c[i]
        desc_list.append(desc_dict.get(code, code))

    if term_count > 1:
        syn_path = os.path.join(EMBEDDING_DIR, f"icd_mimic3_{sort_method}_sort.json")
        if os.path.exists(syn_path):
            with open(syn_path, "r") as f:
                icd_syn = ujson.load(f)
        else:
            icd_syn = {}

        c_desc_list = []
        for i in sorted(ind2c.keys()):
            code = ind2c[i]
            tmp_desc = [desc_list[i]]
            new_terms = icd_syn.get(code, [])
            if len(new_terms) >= term_count - 1:
                tmp_desc.extend(new_terms[: term_count - 1])
            else:
                tmp_desc.extend(new_terms)
                repeat_count = int(term_count / len(tmp_desc)) + 1
                tmp_desc = (tmp_desc * repeat_count)[:term_count]
            c_desc_list.extend(tmp_desc)
    else:
        c_desc_list = desc_list

    pad_id = word2id["**PAD**"]
    unk_id = word2id["**UNK**"]

    c_input_word, c_word_mask = [], []
    for desc in c_desc_list:
        toks = tokenize(desc)
        ids = [word2id.get(w, unk_id) for w in toks]
        m = [1] * len(ids)
        if len(ids) > truncate_length:
            ids = ids[:truncate_length]
            m = m[:truncate_length]
        else:
            pad_len = truncate_length - len(ids)
            ids = ids + [pad_id] * pad_len
            m = m + [0] * pad_len
        c_input_word.append(ids)
        c_word_mask.append(m)

    c_input_word = torch.LongTensor(c_input_word).to(device)
    c_word_mask = torch.LongTensor(c_word_mask).to(device)
    return c_input_word, c_word_mask


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model(checkpoint_path: str, vocab_path: str, device_str: str):
    device = torch.device(device_str)

    # Load checkpoint (entire model was saved via torch.save(model, ...))
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    # Build vocab & label features
    word2id, id2word = load_vocab(vocab_path)
    ind2c = load_ind2c()

    c_input_word, c_word_mask = prepare_label_features(
        word2id, ind2c, device, truncate_length=30, term_count=8, sort_method="random"
    )
    model.c_input_word = c_input_word
    model.c_word_mask = c_word_mask

    # Pre-compute label hidden features once
    with torch.no_grad():
        model.calculate_label_hidden()

    return model, word2id, id2word, ind2c


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def run_inference(model, input_ids, mask, device):
    """Return logits (1×num_codes) and attention weights (1×num_codes×seq_len×heads)."""
    model.decoder._return_attention = True

    input_word = torch.LongTensor([input_ids]).to(device)
    word_mask = torch.LongTensor([mask]).to(device)

    with torch.no_grad():
        hidden = model.calculate_text_hidden(input_word, word_mask)
        logits = model.decoder(hidden, word_mask, model.label_feats)

    alpha = model.decoder._last_alpha  # (1, num_codes, seq_len, heads)
    model.decoder._return_attention = False

    return logits, alpha


# ---------------------------------------------------------------------------
# Integrated Gradients via Captum
# ---------------------------------------------------------------------------


def compute_integrated_gradients(model, input_ids, mask, target_label_idx, device, n_steps=50):
    """
    Compute IG attributions for a single target label.
    Returns a 1-D numpy array of length = number of real tokens.
    """
    from captum.attr import LayerIntegratedGradients

    input_word = torch.LongTensor([input_ids]).to(device)
    word_mask = torch.LongTensor([mask]).to(device)

    # Forward function for captum: takes word embeddings, returns scalar logit for target label
    def forward_from_embeddings(word_embeds):
        # Bypass the embedding layer — feed embeddings directly into combiner
        h = model.encoder.combiner(word_embeds, word_mask)
        logit = model.decoder(h, word_mask, model.label_feats)
        return logit[0, target_label_idx].unsqueeze(0)

    lig = LayerIntegratedGradients(forward_from_embeddings, None)

    # Get the actual embeddings for the input
    word_embeds = model.encoder.word_encoder.word_embedding(input_word)
    word_embeds.requires_grad_(True)

    # Baseline: pad embedding repeated
    pad_id = 0  # We'll use zeros as baseline
    baseline = torch.zeros_like(word_embeds)

    # Use a simple wrapper that takes embeddings directly
    def forward_fn(embeds):
        dropped = model.encoder.word_encoder.word_dropout(embeds)
        h = model.encoder.combiner(dropped, word_mask)
        logit = model.decoder(h, word_mask, model.label_feats)
        return logit[0, target_label_idx].unsqueeze(0)

    # Manual IG computation (more robust on MPS than captum internals)
    attributions = _manual_ig(forward_fn, word_embeds, baseline, n_steps=n_steps)

    # Sum across embedding dim → per-token attribution
    token_attr = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
    seq_len = int(torch.LongTensor(mask).sum().item())
    return token_attr[:seq_len]


def _manual_ig(forward_fn, inputs, baselines, n_steps=50):
    """Trapezoidal-rule Integrated Gradients."""
    # Generate scaled inputs
    scaled_inputs = [
        baselines + (float(i) / n_steps) * (inputs - baselines)
        for i in range(n_steps + 1)
    ]
    # Compute gradients at each step
    grads = []
    for scaled in scaled_inputs:
        scaled = scaled.detach().requires_grad_(True)
        out = forward_fn(scaled)
        out.backward()
        grads.append(scaled.grad.detach().clone())

    # Trapezoidal rule
    grads = torch.stack(grads, dim=0)  # (n_steps+1, ...)
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2.0
    # Actually simpler: just average all
    avg_grads = grads.mean(dim=0)

    ig = (inputs - baselines).detach() * avg_grads
    return ig


# ---------------------------------------------------------------------------
# HTML highlighting
# ---------------------------------------------------------------------------


def highlight_html(tokens, scores, title=""):
    """
    Build an HTML string where each token is colored by its normalised score.
    scores: array of shape (num_tokens,) — higher = more important.
    """
    if len(scores) == 0:
        return "<p>No scores available</p>"

    # Normalise scores to [0, 1]
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        normed = np.zeros_like(scores)
    else:
        normed = (scores - mn) / (mx - mn)

    html_parts = []
    if title:
        html_parts.append(f"<h4>{title}</h4>")
    html_parts.append('<div style="line-height:1.8; font-family: monospace; font-size:14px;">')

    for tok, s in zip(tokens, normed):
        # Interpolate from white (0) to red (1)
        r = 255
        g = int(255 * (1 - s))
        b = int(255 * (1 - s))
        bg = f"rgb({r},{g},{b})"
        html_parts.append(
            f'<span style="background-color:{bg}; padding:2px 3px; margin:1px; '
            f'border-radius:3px; display:inline-block;" '
            f'title="{s:.4f}">{tok}</span> '
        )
    html_parts.append("</div>")
    return "".join(html_parts)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="MSMN Interpretability Demo", layout="wide")
    st.title("🏥 MSMN ICD Coding — Interpretability Demo")

    # ---- Sidebar: config ----
    with st.sidebar:
        st.header("Configuration")
        checkpoint_path = st.text_input(
            "Model checkpoint path",
            value=MSMN_MODEL_PATH,
        )
        vocab_path = st.text_input(
            "Word embedding path (gensim .model)",
            value=EMBEDDING_MODEL_PATH,
        )
        ig_steps = st.slider("IG steps", min_value=10, max_value=100, value=30, step=10)
        threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)
        device = get_device()
        st.info(f"Device: **{device}**")

    # ---- Load model ----
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found at `{checkpoint_path}`. Please download the mimic3-50 checkpoint and place it there.")
        st.stop()
    if not os.path.exists(vocab_path):
        st.error(f"Word embedding not found at `{vocab_path}`. Please download word2vec_sg0_100.model and place it there.")
        st.stop()

    with st.spinner("Loading model…"):
        model, word2id, id2word, ind2c = load_model(checkpoint_path, vocab_path, str(device))

    desc_dict = load_code_descriptions()

    # ---- Input text ----
    st.subheader("📝 Input Medical Note")
    note_text = st.text_area("Edit or paste a medical note:", value=SYNTHETIC_NOTE, height=300)

    if st.button("🔍 Run Inference", type="primary"):
        tokens, input_ids, mask = text_to_input(note_text, word2id)
        seq_len = len(tokens)

        if seq_len == 0:
            st.warning("No valid tokens found in the note.")
            st.stop()

        # Run forward pass
        with st.spinner("Running forward pass…"):
            logits, alpha = run_inference(model, input_ids, mask, device)

        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (num_codes,)

        # Get predicted codes above threshold
        pred_indices = np.where(probs >= threshold)[0]

        if len(pred_indices) == 0:
            st.warning("No ICD codes predicted above threshold. Try lowering the threshold.")
            # Still show top-5
            pred_indices = np.argsort(probs)[-5:][::-1]
            st.info("Showing top-5 codes by probability instead.")

        # Build selection list
        code_options = []
        for idx in sorted(pred_indices, key=lambda i: -probs[i]):
            code = ind2c.get(idx, f"UNK_{idx}")
            desc = desc_dict.get(code, "")
            prob = probs[idx]
            code_options.append((idx, code, desc, prob))

        st.subheader("🏷️ Predicted ICD Codes")
        option_labels = [
            f"{code} — {desc[:60]}{'…' if len(desc)>60 else ''} (p={prob:.3f})"
            for _, code, desc, prob in code_options
        ]
        selected = st.selectbox("Select a code to visualise:", option_labels)
        sel_idx = option_labels.index(selected)
        target_label_idx = code_options[sel_idx][0]
        target_code = code_options[sel_idx][1]
        target_desc = code_options[sel_idx][2]
        target_prob = code_options[sel_idx][3]

        st.markdown(f"**Selected:** `{target_code}` — {target_desc} (probability: {target_prob:.4f})")

        # ---- Attention scores for selected code ----
        # alpha shape: (1, num_codes, seq_len, heads) — average over heads
        alpha_np = alpha.squeeze(0).cpu().numpy()  # (num_codes, seq_len, heads)
        attn_scores = alpha_np[target_label_idx, :seq_len, :].mean(axis=-1)  # (seq_len,)

        # ---- Integrated Gradients ----
        with st.spinner(f"Computing Integrated Gradients ({ig_steps} steps)…"):
            ig_scores = compute_integrated_gradients(
                model, input_ids, mask, target_label_idx, device, n_steps=ig_steps
            )
        # Take absolute value — magnitude of attribution
        ig_scores_abs = np.abs(ig_scores)

        # ---- Side-by-side visualisation ----
        st.subheader(f"🔬 Interpretability for `{target_code}`")
        col1, col2 = st.columns(2)

        with col1:
            html_attn = highlight_html(tokens, attn_scores, title="Attention Scores")
            st.markdown(html_attn, unsafe_allow_html=True)

        with col2:
            html_ig = highlight_html(tokens, ig_scores_abs, title="Integrated Gradients (|attr|)")
            st.markdown(html_ig, unsafe_allow_html=True)

        # ---- All predicted codes table ----
        st.subheader("📋 All Predicted Codes")
        
        df = pd.DataFrame(
            [(code, desc, f"{prob:.4f}") for _, code, desc, prob in code_options],
            columns=["ICD Code", "Description", "Probability"],
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Store in session so user can switch codes without re-running
        st.session_state["tokens"] = tokens
        st.session_state["input_ids"] = input_ids
        st.session_state["mask"] = mask
        st.session_state["alpha"] = alpha
        st.session_state["probs"] = probs
        st.session_state["code_options"] = code_options


if __name__ == "__main__":
    main()
