"""Utility functions for MSMN inference and interpretability (no Streamlit dependency)."""

import os
import re
import csv
import torch
import numpy as np
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import gensim
import ujson
from captum.attr import LayerIntegratedGradients

from constant import DATA_DIR, MIMIC_3_DIR, EMBEDDING_DIR

_tokenizer = RegexpTokenizer(r"\w+")


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
# Vocabulary helpers
# ---------------------------------------------------------------------------


def load_vocab(vocab_path: str):
    """Build word2id / id2word from a gensim word2vec .model file.
    """
    

    _orig_np_load = np.load

    def _safe_np_load(fname, mmap_mode=None, allow_pickle=False,
                      fix_imports=True, encoding="ASCII"):
        try:
            return _orig_np_load(fname, mmap_mode=mmap_mode,
                                 allow_pickle=allow_pickle,
                                 fix_imports=fix_imports, encoding=encoding)
        except (FileNotFoundError, IOError):
            # Return a placeholder zero array
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


def load_code_descriptions():
    desc_dict = defaultdict(str)
    diag_path = os.path.join(DATA_DIR, "D_ICD_DIAGNOSES.csv")
    proc_path = os.path.join(DATA_DIR, "D_ICD_PROCEDURES.csv")
    icd9_path = os.path.join(DATA_DIR, "ICD9_descriptions")

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
# Build ind2c for mimic3-50
# ---------------------------------------------------------------------------


def load_ind2c():
    """Reproduce the sorted code→index mapping used during training."""
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
# Tokenisation (mirrors MimicFullDataset)
# ---------------------------------------------------------------------------


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
# Label feature preparation
# ---------------------------------------------------------------------------


def prepare_label_features(word2id, ind2c, device, truncate_length=30,
                            term_count=8, sort_method="random"):
    """Replicate MimicFullDataset.prepare_label_feature for mimic3-50."""

    desc_dict = load_code_descriptions()
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
# Inference
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
# Integrated Gradients
# ---------------------------------------------------------------------------


def compute_integrated_gradients(model, input_ids, mask, target_label_idx,
                                  device, n_steps=100):
    """
    Compute IG attributions for a single target label using Captum.
    Returns a 1-D numpy array of length = number of real tokens.
    """
    
    input_word = torch.LongTensor([input_ids]).to(device)
    word_mask = torch.LongTensor([mask]).to(device)

    # Forward function: takes input word IDs, returns scalar logit for target label
    def forward_fn(input_ids_tensor):
        hidden = model.calculate_text_hidden(input_ids_tensor, word_mask)
        logits = model.decoder(hidden, word_mask, model.label_feats)
        return logits[:, target_label_idx]

    # Use Captum's LayerIntegratedGradients on the embedding layer
    lig = LayerIntegratedGradients(forward_fn, model.encoder.word_encoder.word_embedding)
    
    # Baseline: use PAD token ID (typically 0 or the actual PAD ID from vocab)
    baseline = torch.zeros_like(input_word)
    
    # Compute attributions
    attributions = lig.attribute(
        input_word,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=n_steps,  # Process all steps in one batch if memory allows
    )
    
    # Sum across embedding dimension -> per-token attribution
    token_attr = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
    seq_len = int(word_mask.sum().item())
    return token_attr[:seq_len]


# ---------------------------------------------------------------------------
# HTML highlighting
# ---------------------------------------------------------------------------


def highlight_html_green(tokens, scores, title="", top_k=0):
    """Build an HTML string with tokens colored white→green by normalised score.
    
    Args:
        tokens: List of token strings
        scores: Array of scores (higher = more important)
        title: Optional title for the visualization
        top_k: If > 0, dim all but the top-k highest scoring tokens
    """
    if len(scores) == 0:
        return "<p>No scores available</p>"

    # Identify top-k indices if filtering is enabled
    if top_k > 0:
        top_k_indices = set(np.argsort(scores)[-top_k:])
    else:
        top_k_indices = None

    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        normed = np.zeros_like(scores)
    else:
        normed = (scores - mn) / (mx - mn)

    html_parts = []
    if title:
        html_parts.append(f"<h4>{title}</h4>")
    html_parts.append('<div style="line-height:1.8; font-family: monospace; font-size:14px;">')

    for idx, (tok, s) in enumerate(zip(tokens, normed)):
        # Check if token is in top-k
        if top_k_indices is not None and idx not in top_k_indices:
            # Non-top-k: no background, plain text
            html_parts.append(
                f'<span style="padding:2px 3px; margin:1px; display:inline-block;">{tok}</span> '
            )
        else:
            # Top-k or no filtering: show with colored background
            # White (low) → green (high)
            r = int(255 * (1 - s))
            g = 255
            b = int(255 * (1 - s))
            bg = f"rgb({r},{g},{b})"
            
            html_parts.append(
                f'<span style="background-color:{bg}; padding:2px 3px; margin:1px; '
                f'border-radius:3px; display:inline-block;" '
                f'title="{s:.4f}">{tok}</span> '
            )
    html_parts.append("</div>")
    return "".join(html_parts)


def highlight_html_diverging(tokens, scores, title="", top_k=0):
    """Build an HTML string with tokens colored red→white→green for signed scores.
    
    Negative scores → red, zero → white, positive scores → green.
    
    Args:
        tokens: List of token strings
        scores: Array of signed scores
        title: Optional title for the visualization
        top_k: If > 0, dim all but the top-k tokens by value
    """
    if len(scores) == 0:
        return "<p>No scores available</p>"

    # Identify top-k indices if filtering is enabled 
    if top_k > 0:
        top_k_indices = set(np.argsort(scores)[-top_k:])
    else:
        top_k_indices = None

    html_parts = []
    if title:
        html_parts.append(f"<h4>{title}</h4>")
    html_parts.append('<div style="line-height:1.8; font-family: monospace; font-size:14px;">')

    # Find symmetric range around zero
    abs_max = max(abs(scores.min()), abs(scores.max()))
    if abs_max < 1e-9:
        abs_max = 1.0

    for idx, (tok, score) in enumerate(zip(tokens, scores)):
        # Check if token is in top-k
        if top_k_indices is not None and idx not in top_k_indices:
            # Non-top-k: no background, plain text
            html_parts.append(
                f'<span style="padding:2px 3px; margin:1px; display:inline-block;">{tok}</span> '
            )
        else:
            # Top-k or no filtering: show with colored background
            # Normalize to [-1, 1]
            normed = score / abs_max
            
            if normed < 0:
                # Negative: white → red
                intensity = abs(normed)
                r = 255
                g = int(255 * (1 - intensity))
                b = int(255 * (1 - intensity))
            else:
                # Positive: white → green
                intensity = normed
                r = int(255 * (1 - intensity))
                g = 255
                b = int(255 * (1 - intensity))
            
            bg = f"rgb({r},{g},{b})"
            
            html_parts.append(
                f'<span style="background-color:{bg}; padding:2px 3px; margin:1px; '
                f'border-radius:3px; display:inline-block;" '
                f'title="{score:.4f}">{tok}</span> '
            )
    html_parts.append("</div>")
    return "".join(html_parts)
