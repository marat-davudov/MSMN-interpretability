"""
Streamlit demo for MSMN model interpretability.
Run with:  streamlit run demo.py
"""

import os
import torch
import numpy as np
import streamlit as st
import pandas as pd
import json
from constant import MSMN_MODEL_PATH, EMBEDDING_MODEL_PATH, IG_STEPS, THRESHOLD, MIMIC_3_DIR
from utils import (
    get_device,
    load_vocab as _load_vocab,
    load_code_descriptions as _load_code_descriptions,
    load_ind2c as _load_ind2c,
    text_to_input,
    prepare_label_features,
    run_inference,
    compute_integrated_gradients,
    highlight_html_green,
    highlight_html_diverging,
)

load_vocab = st.cache_resource(_load_vocab)
load_code_descriptions = st.cache_resource(_load_code_descriptions)
load_ind2c = st.cache_resource(_load_ind2c)

NOTE_EXAMPLES = json.load(open(os.path.join(MIMIC_3_DIR, "mimic3-50_test.json"), "r"))
SYNTHETIC_NOTE = NOTE_EXAMPLES[0]["TEXT"]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model(checkpoint_path: str, vocab_path: str, device_str: str):
    device = torch.device(device_str)

    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    word2id, id2word = load_vocab(vocab_path)
    ind2c = load_ind2c()

    c_input_word, c_word_mask = prepare_label_features(
        word2id, ind2c, device, truncate_length=30, term_count=8, sort_method="random"
    )
    model.c_input_word = c_input_word
    model.c_word_mask = c_word_mask

    with torch.no_grad():
        model.calculate_label_hidden()

    return model, word2id, id2word, ind2c



# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="MSMN Interpretability Demo",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None,
    )
    # Force light theme 
    st.markdown("""
        <style>
        :root {
            color-scheme: light;
        }
        </style>
    """, unsafe_allow_html=True)
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
        pred_indices = np.where(probs >= THRESHOLD)[0]
        fallback = False
        if len(pred_indices) == 0:
            pred_indices = np.argsort(probs)[-5:][::-1]
            fallback = True

        code_options = []
        for idx in sorted(pred_indices, key=lambda i: -probs[i]):
            code = ind2c.get(idx, f"UNK_{idx}")
            desc = desc_dict.get(code, "")
            prob = probs[idx]
            code_options.append((idx, code, desc, prob))

        # Preserve previously selected code across re-runs if it still appears
        prev_code = st.session_state.get("selected_code")
        new_codes = [c for _, c, _, _ in code_options]
        default_idx = new_codes.index(prev_code) if prev_code in new_codes else 0

        # Pre-compute IG for all predicted codes upfront
        ig_cache = {}
        progress_bar = st.progress(0, text=f"Computing Integrated Gradients for {len(code_options)} codes ({IG_STEPS} steps each)…")
        for i, (idx, code, desc, prob) in enumerate(code_options):
            ig_scores = compute_integrated_gradients(
                model, input_ids, mask, idx, device, n_steps=IG_STEPS
            )
            ig_cache[idx] = ig_scores
            progress_bar.progress((i + 1) / len(code_options), text=f"Computing IG for code {i+1}/{len(code_options)}: {code}")
        progress_bar.empty()

        st.session_state.update({
            "tokens": tokens,
            "input_ids": input_ids,
            "mask": mask,
            "alpha": alpha,
            "probs": probs,
            "code_options": code_options,
            "fallback": fallback,
            "sel_idx": default_idx,
            "ig_cache": ig_cache,
        })

    # ---- Results (shown whenever session state has data) ----
    if "code_options" not in st.session_state:
        st.stop()

    code_options = st.session_state["code_options"]
    tokens = st.session_state["tokens"]
    input_ids = st.session_state["input_ids"]
    mask = st.session_state["mask"]
    alpha = st.session_state["alpha"]
    seq_len = sum(mask[:len(tokens)])

    if st.session_state.get("fallback"):
        st.warning("No ICD codes predicted above threshold — showing top-5 by probability.")

    st.subheader("🏷️ Predicted ICD Codes")
    option_labels = [
        f"{code} — {desc[:60]}{'…' if len(desc)>60 else ''} (p={prob:.3f})"
        for _, code, desc, prob in code_options
    ]

    sel_idx = st.selectbox(
        "Select a code to visualise:",
        range(len(option_labels)),
        format_func=lambda i: option_labels[i],
        index=st.session_state.get("sel_idx", 0),
        key="code_selectbox",
    )
    # Persist selection so re-runs after inference keep the same code
    st.session_state["sel_idx"] = sel_idx
    st.session_state["selected_code"] = code_options[sel_idx][1]

    target_label_idx, target_code, target_desc, target_prob = code_options[sel_idx]
    st.markdown(f"**Selected:** `{target_code}` — {target_desc} (probability: {target_prob:.4f})")

    # ---- Top-K filter control ----
    top_k = st.slider(
        "Show top-K most important tokens (0 = show all)",
        min_value=5,
        max_value=min(50, len(tokens)),
        value=0,
        help="Filter to show only the K tokens with highest scores. Set to 0 to show all tokens."
    )

    # ---- Attention scores for selected code ----
    alpha_np = alpha.squeeze(0).cpu().numpy()  # (num_codes, seq_len, heads)
    attn_scores = alpha_np[target_label_idx, :seq_len, :].mean(axis=-1)  # (seq_len,)

    # ---- Integrated Gradients (pre-computed for all codes) ----
    ig_cache = st.session_state["ig_cache"]
    ig_scores = ig_cache[target_label_idx]  # Keep signed values for red-green coloring

    # ---- Side-by-side visualisation ----
    st.subheader(f"🔬 Interpretability for `{target_code}`")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(highlight_html_green(tokens, attn_scores, title="Attention Scores", top_k=top_k), unsafe_allow_html=True)
    with col2:
        st.markdown(highlight_html_diverging(tokens, ig_scores, title="Integrated Gradients", top_k=top_k), unsafe_allow_html=True)

    # ---- All predicted codes table ----
    st.subheader("📋 All Predicted Codes")
    df = pd.DataFrame(
        [(code, desc, f"{prob:.4f}") for _, code, desc, prob in code_options],
        columns=["ICD Code", "Description", "Probability"],
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
