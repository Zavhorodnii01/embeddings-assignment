
import os
import warnings
import json
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")   # saves figures to file without opening windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import random

random.seed(42)
np.random.seed(42)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
os.makedirs(OUT, exist_ok=True)

CORPUS_RAW = [
    "the bank approved the mortgage application", "she deposited money at the bank",
    "the bank manager called about the loan", "he withdrew cash from the bank",
    "the bank offers low interest rates", "the central bank raised interest rates",
    "the investment bank advised the merger", "the bank statement shows all transactions",
    "she applied for a credit card at the bank", "the bank closed her account",
    "the bank teller processed the cheque", "the savings bank pays higher interest",
    "the commercial bank offers business loans", "he transferred funds between bank accounts",
    "the bank vault stores gold reserves", "her bank balance increased significantly",
    "the bank declined the loan request", "the regional bank opened new branches",
    "online banking replaced visits to the bank", "the bank reported record profits",
    "we sat on the river bank watching fish", "the flood destroyed the bank of the river",
    "wildflowers grew along the bank of the stream", "children played on the muddy river bank",
    "the bank of the canal was overgrown with reeds", "the fisherman sat on the steep river bank",
    "erosion is destroying the river bank", "the bank of the lake is sandy and calm",
    "they built a path along the bank of the river", "the river bank flooded after heavy rain",
    "moss covered the bank beside the stream", "the bank collapsed into the water",
    "we camped on the grassy bank near the river", "the bank of the creek offers shade",
    "the steep bank made fishing difficult", "deer drank from the bank of the stream",
    "the bank of the river was frozen in winter", "trees lined the opposite bank of the river",
    "the bank of the lake is protected wetland", "she walked along the bank of the river",
    "the patient received a hospital discharge after surgery", "discharge papers were signed by the doctor",
    "the nurse prepared the discharge summary", "early discharge is recommended for stable patients",
    "hospital discharge planning reduces readmission", "the patient was given a discharge diagnosis",
    "the doctor arranged for home care after discharge", "discharge instructions were given to the family",
    "premature discharge can be dangerous", "the discharge summary listed all medications",
    "the patient refused discharge from the ward", "delayed discharge leads to bed shortages",
    "the discharge letter was sent to the general practitioner", "post discharge follow up is essential",
    "the patient was awaiting discharge for two days", "elective discharge occurred on monday morning",
    "the electrical discharge caused a fire", "static discharge damaged the circuit board",
    "lightning is a form of electrical discharge", "the capacitor discharge was measured in microseconds",
    "the battery discharge rate affects performance", "arc discharge produces intense light",
    "corona discharge occurs near high voltage lines", "the discharge of the capacitor was rapid",
    "the electrical discharge welded the metals together", "plasma discharge glows blue",
    "a sudden discharge of electricity occurred", "the discharge of static electricity shocked him",
    "electrostatic discharge can destroy microchips", "the discharge rate of the battery was low",
    "the high voltage discharge was visible", "the discharge of energy lit up the room",
    "lead exposure causes serious neurological damage", "children are most vulnerable to lead poisoning",
    "lead paint was banned due to health risks", "high blood lead levels impair cognition",
    "lead is a heavy toxic metal", "environmental lead contamination is widespread",
    "the plumbing pipes contained lead", "lead accumulates in bones and soft tissue",
    "the factory released lead into the river", "testing for lead exposure is recommended",
    "the child had elevated blood lead levels", "lead poisoning symptoms include fatigue",
    "drinking water contaminated with lead is dangerous", "the old paint chips contained lead",
    "workers were exposed to lead dust", "lead shields protect against radiation",
    "the doctor attached the lead to the patient chest", "the twelve lead ECG showed abnormalities",
    "the cardiologist examined the precordial lead", "lead two showed st elevation",
    "the ECG lead fell off during the procedure", "the nurse placed the chest lead correctly",
    "the augmented lead avr showed changes", "the limb lead detected arrhythmia",
    "the ECG lead placement follows standard protocol", "the precordial lead showed right bundle branch block",
    "the lead wire connects the electrode to the monitor", "an ECG lead records electrical activity",
    "the weather is very cold today", "she wrapped herself in a blanket because it was cold",
    "the cold wind made walking unpleasant", "the river water was ice cold in winter",
    "cold temperatures froze the pipes", "the cold snap lasted for a week",
    "the mountain peak was bitterly cold", "they stayed indoors during the cold weather",
    "cold air masses moved from the north", "the cold front brought heavy snowfall",
    "the cold winter reduced crop yields", "the cold climate requires special clothing",
    "the cold ocean current affects the local climate", "the night was extremely cold",
    "cold storage keeps food fresh", "the cold snap killed the crops",
    "she caught a cold last week", "the cold kept him home from work",
    "the common cold is caused by rhinovirus", "he had a runny nose from the cold",
    "the cold spread quickly through the office", "she recovered from the cold in three days",
    "cold symptoms include sneezing and sore throat", "the cold made her throat painful",
    "the doctor said it was just a cold", "vitamin c may shorten the duration of a cold",
    "the cold virus spreads through droplets", "he developed a cold after the conference",
    "the cold left him with a persistent cough", "the cold season peaks in winter months",
    "the cold spread from her child to the whole family", "antibiotics do not treat a cold",
    "the doctor examined the patient carefully", "the nurse administered the medication",
    "the hospital admitted ten new patients", "the surgeon performed the operation successfully",
    "the laboratory analyzed the blood sample", "the diagnosis was confirmed by imaging",
    "the treatment plan included physical therapy", "the patient reported severe pain",
    "the medical record was updated immediately", "the pharmacy dispensed the prescription",
    "the clinical trial enrolled fifty volunteers", "the immune system fights infection",
    "the cardiovascular system pumps blood", "the nervous system transmits signals",
    "the respiratory system provides oxygen", "the endocrine system regulates hormones",
    "the digestive system processes nutrients", "the skeletal system supports the body",
    "the musculoskeletal system enables movement", "the renal system filters waste",
    "the scientist published important research", "the student studied late into the night",
    "the teacher explained the concept clearly", "the engineer designed a new system",
    "the computer program processed the data", "the algorithm found the optimal solution",
    "the machine learning model improved accuracy", "the neural network trained on large data",
    "the researcher analyzed the experimental results", "the paper was accepted for publication",
    "the model predicted the outcome accurately", "the data showed a significant trend",
    "the experiment confirmed the hypothesis", "the results were statistically significant",
    "the method outperformed the baseline significantly", "the evaluation metric was precision and recall",
]

CORPUS = [s.lower().split() for s in CORPUS_RAW]

print(f"Corpus: {len(CORPUS)} sentences, {sum(len(s) for s in CORPUS)} tokens")

print("\n--- Training Word2Vec ---")
w2v = Word2Vec(
    sentences=CORPUS,
    vector_size=64,
    window=5,
    min_count=1,
    workers=2,
    epochs=200,
    seed=42,
)
print(f"Vocabulary size: {len(w2v.wv)}")

def w2v_vec(word):
    return w2v.wv[word]

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print("\n--- Building Transformer ---")

DIM = 64   # must match word2vec dim for fair comparison
NUM_HEADS = 4
HEAD_DIM = DIM // NUM_HEADS

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)
    return PE

rng = np.random.RandomState(0)
Wq = rng.randn(DIM, DIM) * 0.1
Wk = rng.randn(DIM, DIM) * 0.1
Wv = rng.randn(DIM, DIM) * 0.1
Wo = rng.randn(DIM, DIM) * 0.1
Wff1 = rng.randn(DIM, DIM * 4) * 0.1
Wff2 = rng.randn(DIM * 4, DIM) * 0.1

def layer_norm(x, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    std = x.std(-1, keepdims=True)
    return (x - mu) / (std + eps)

def multi_head_attention(X):
    """X: (seq_len, dim) → (seq_len, dim), weights: (seq_len, seq_len)"""
    seq_len = X.shape[0]
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    Q_h = Q.reshape(seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)  # (h, s, d)
    K_h = K.reshape(seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)
    V_h = V.reshape(seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)
    scores = Q_h @ K_h.transpose(0, 2, 1) / np.sqrt(HEAD_DIM)  # (h, s, s)
    weights = softmax(scores, axis=-1)
    attn = weights @ V_h  # (h, s, d)
    attn = attn.transpose(1, 0, 2).reshape(seq_len, DIM)  # (s, dim)
    out = attn @ Wo
    avg_weights = weights.mean(axis=0)
    return out, avg_weights

def feed_forward(x):
    return np.maximum(0, x @ Wff1) @ Wff2

def transformer_encode(sentence_tokens):
    """
    sentence_tokens: list of str
    Returns (contextual_embeddings, attention_weights)
    contextual_embeddings: (seq_len, DIM)
    """
    vecs = []
    for tok in sentence_tokens:
        if tok in w2v.wv:
            vecs.append(w2v.wv[tok])
        else:
            vecs.append(rng.randn(DIM) * 0.01)
    X = np.array(vecs)                           # (seq, DIM)
    X = X + positional_encoding(len(X), DIM)
    X = layer_norm(X)
    attn_out, attn_weights = multi_head_attention(X)
    X = layer_norm(X + attn_out)
    ff_out = feed_forward(X)
    X = layer_norm(X + ff_out)
    return X, attn_weights

def get_contextual_vec(sentence_tokens, target_word):
    embs, weights = transformer_encode(sentence_tokens)
    idx = next((i for i, t in enumerate(sentence_tokens) if t == target_word), None)
    if idx is None:
        return None, None, None
    return embs[idx], weights[idx], idx

examples = {
    "bank": {
        "financial": [
            "the bank approved the mortgage application",
            "she deposited money at the bank",
            "the bank manager called about the loan",
        ],
        "river": [
            "we sat on the river bank watching fish",
            "the bank of the canal was overgrown with reeds",
            "the steep bank made fishing difficult",
        ],
    },
    "discharge": {
        "medical": [
            "the patient received a hospital discharge after surgery",
            "discharge papers were signed by the doctor",
            "the nurse prepared the discharge summary",
        ],
        "electrical": [
            "the electrical discharge caused a fire",
            "the capacitor discharge was measured in microseconds",
            "lightning is a form of electrical discharge",
        ],
    },
    "lead": {
        "toxic": [
            "lead exposure causes serious neurological damage",
            "high blood lead levels impair cognition",
            "the child had elevated blood lead levels",
        ],
        "ecg": [
            "the doctor attached the lead to the patient chest",
            "the twelve lead ECG showed abnormalities",
            "the cardiologist examined the precordial lead",
        ],
    },
    "cold": {
        "temperature": [
            "the weather is very cold today",
            "cold temperatures froze the pipes",
            "the cold wind made walking unpleasant",
        ],
        "illness": [
            "she caught a cold last week",
            "the cold kept him home from work",
            "the common cold is caused by rhinovirus",
        ],
    },
}

print("\n--- Computing contextual embeddings ---")

results = {}
for word, contexts in examples.items():
    results[word] = {}
    for ctx_name, sentences in contexts.items():
        w2v_vecs = []
        tf_vecs  = []
        for sent in sentences:
            tokens = sent.lower().split()
            if word in tokens and word in w2v.wv:
                w2v_vecs.append(w2v_vec(word))
            tv, _, _ = get_contextual_vec(tokens, word)
            if tv is not None:
                tf_vecs.append(tv)
        results[word][ctx_name] = {
            "w2v": np.array(w2v_vecs),
            "tf":  np.array(tf_vecs),
        }

print("--- Figure 1: Cosine similarity heatmap ---")

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Cosine Similarity: same word in different contexts\n"
             "Left: Word2Vec (static)   |   Right: Transformer (contextual)",
             fontsize=13, fontweight="bold", y=1.01)

for col_i, (word, contexts) in enumerate(results.items()):
    ctx_names = list(contexts.keys())
    for row_i, model_key in enumerate(["w2v", "tf"]):
        ax = axes[row_i][col_i]
        all_vecs = []
        labels   = []
        for ctx in ctx_names:
            vecs = contexts[ctx][model_key]
            for j, v in enumerate(vecs):
                all_vecs.append(v)
                labels.append(f"{ctx[:3]}-{j+1}")
        mat = cosine_similarity(np.array(all_vecs))
        n = len(contexts[ctx_names[0]][model_key])
        for boundary in range(n, len(all_vecs), n):
            ax.axhline(boundary - 0.5, color="white", lw=2)
            ax.axvline(boundary - 0.5, color="white", lw=2)
        sns.heatmap(
            mat, ax=ax,
            vmin=0, vmax=1,
            cmap="RdYlGn",
            annot=True, fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            annot_kws={"size": 7},
        )
        model_label = "Word2Vec" if model_key == "w2v" else "Transformer"
        ax.set_title(f'"{word}"\n{model_label}', fontsize=9, fontweight="bold")
        ax.tick_params(axis="both", labelsize=6, rotation=45)

plt.tight_layout()
plt.savefig(f"{OUT}fig1_similarity_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig1")

print("--- Figure 2: Cross-context similarity bars ---")

cross_sims_w2v = {}
cross_sims_tf  = {}

for word, contexts in results.items():
    ctx_names = list(contexts.keys())
    ctx_a, ctx_b = ctx_names[0], ctx_names[1]
    vecs_a_w2v = contexts[ctx_a]["w2v"]
    vecs_b_w2v = contexts[ctx_b]["w2v"]
    vecs_a_tf  = contexts[ctx_a]["tf"]
    vecs_b_tf  = contexts[ctx_b]["tf"]
    mu_a_w2v = vecs_a_w2v.mean(0)
    mu_b_w2v = vecs_b_w2v.mean(0)
    mu_a_tf  = vecs_a_tf.mean(0)
    mu_b_tf  = vecs_b_tf.mean(0)
    cross_sims_w2v[word] = cosine_sim(mu_a_w2v, mu_b_w2v)
    cross_sims_tf[word]  = cosine_sim(mu_a_tf,  mu_b_tf)

fig, ax = plt.subplots(figsize=(8, 5))
words = list(cross_sims_w2v.keys())
x = np.arange(len(words))
w = 0.35
bars1 = ax.bar(x - w/2, [cross_sims_w2v[wd] for wd in words], w,
               label="Word2Vec", color="#E07B54", edgecolor="white")
bars2 = ax.bar(x + w/2, [cross_sims_tf[wd]  for wd in words], w,
               label="Transformer", color="#5B9BD5", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f'"{wd}"' for wd in words], fontsize=12)
ax.set_ylabel("Cosine similarity (cross-context)", fontsize=11)
ax.set_title("Cross-context similarity for polysemous words\n"
             "(lower = model better distinguishes the two meanings)", fontsize=12)
ax.set_ylim(0, 1.1)
ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6)
ax.legend(fontsize=11)
for bar in bars1:
    ax.annotate(f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}fig2_cross_context_similarity.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig2")

print("--- Figure 3: Attention weights ---")

bank_sentences = {
    "Financial context": "the bank approved the mortgage application",
    "River context":     "we sat on the river bank watching fish",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Self-attention weights for "bank" in two different contexts\n'
             '(highlights which tokens the model attends to when encoding "bank")',
             fontsize=12, fontweight="bold")

for ax, (ctx_label, sent) in zip(axes, bank_sentences.items()):
    tokens = sent.lower().split()
    _, attn_w, idx = get_contextual_vec(tokens, "bank")
    bar_colors = ["#E07B54" if i == idx else "#5B9BD5" for i in range(len(tokens))]
    ax.bar(range(len(tokens)), attn_w, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Attention weight", fontsize=10)
    ax.set_title(ctx_label, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(attn_w) * 1.3)
    ax.get_xticklabels()[idx].set_color("#E07B54")
    ax.get_xticklabels()[idx].set_fontweight("bold")

plt.tight_layout()
plt.savefig(f"{OUT}fig3_attention_weights.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig3")

print("--- Figure 4: PCA visualisation ---")

pca_words = ["discharge", "lead", "bank", "cold"]
pca_sentences = {
    "discharge_medical":    "the patient received a hospital discharge after surgery",
    "discharge_electrical": "the electrical discharge caused a fire",
    "lead_toxic":           "lead exposure causes serious neurological damage",
    "lead_ecg":             "the doctor attached the lead to the patient chest",
    "bank_financial":       "the bank approved the mortgage application",
    "bank_river":           "we sat on the river bank watching fish",
    "cold_illness":         "she caught a cold last week",
    "cold_temperature":     "the weather is very cold today",
}

w2v_pca_vecs  = []
tf_pca_vecs   = []
pca_labels    = []
pca_colors_w2v = []
pca_colors_tf  = []
color_map = {
    "discharge": "#E07B54",
    "lead":      "#5B9BD5",
    "bank":      "#70AD47",
    "cold":      "#9B59B6",
}
marker_map = {"medical": "o", "electrical": "^", "toxic": "o", "ecg": "^",
              "financial": "o", "river": "^", "illness": "o", "temperature": "^"}

for label, sent in pca_sentences.items():
    word = label.split("_")[0]
    ctx  = label.split("_")[1]
    tokens = sent.lower().split()
    if word in tokens and word in w2v.wv:
        w2v_pca_vecs.append(w2v_vec(word))
    else:
        w2v_pca_vecs.append(np.zeros(DIM))
    tv, _, _ = get_contextual_vec(tokens, word)
    tf_pca_vecs.append(tv if tv is not None else np.zeros(DIM))
    pca_labels.append(label)
    pca_colors_w2v.append(color_map[word])
    pca_colors_tf.append(color_map[word])

pca = PCA(n_components=2, random_state=42)
w2v_2d = pca.fit_transform(np.array(w2v_pca_vecs))
pca2   = PCA(n_components=2, random_state=42)
tf_2d  = pca2.fit_transform(np.array(tf_pca_vecs))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("PCA of word representations for polysemous medical/general terms",
             fontsize=12, fontweight="bold")

for ax, coords, title in [(ax1, w2v_2d, "Word2Vec (static)"), (ax2, tf_2d, "Transformer (contextual)")]:
    for i, lbl in enumerate(pca_labels):
        word = lbl.split("_")[0]
        ctx  = lbl.split("_")[1]
        marker = "o" if ctx in ("medical", "toxic", "financial", "illness") else "^"
        ax.scatter(coords[i, 0], coords[i, 1],
                   color=color_map[word], marker=marker, s=120,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(lbl.replace("_", "\n"),
                    (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    words_to_connect = pca_words
    for wd in words_to_connect:
        idxs = [j for j, l in enumerate(pca_labels) if l.startswith(wd + "_")]
        if len(idxs) == 2:
            ax.plot([coords[idxs[0], 0], coords[idxs[1], 0]],
                    [coords[idxs[0], 1], coords[idxs[1], 1]],
                    color=color_map[wd], lw=1.5, ls="--", alpha=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.grid(True, alpha=0.3)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=color_map[wd],
           markeredgecolor="black", markersize=9, label=wd)
    for wd in pca_words
]
legend_elements += [
    Line2D([0],[0], marker="o", color="gray", ls="", label="context A"),
    Line2D([0],[0], marker="^", color="gray", ls="", label="context B"),
]
ax2.legend(handles=legend_elements, fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT}fig4_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig4")

print("--- Figure 5: Sentence similarity ---")

sentence_pairs = [
    ("patient has high fever",      "patient has elevated temperature",  "paraphrase (medical)"),
    ("the bank closed early today", "the financial institution shut down", "paraphrase (finance)"),
    ("the patient recovered fast",  "the bank is by the river",           "unrelated"),
    ("lead poisoning is dangerous", "toxic metal exposure harms children", "paraphrase (toxicology)"),
    ("cold virus spreads by air",   "rhinovirus causes common illness",   "paraphrase (virology)"),
    ("doctor signed discharge form","nurse left the hospital building",   "partial overlap"),
]

def sentence_w2v_vec(sent):
    tokens = [t for t in sent.lower().split() if t in w2v.wv]
    if not tokens:
        return np.zeros(DIM)
    return np.mean([w2v_vec(t) for t in tokens], axis=0)

def sentence_tf_vec(sent):
    tokens = sent.lower().split()
    embs, _ = transformer_encode(tokens)
    return embs.mean(0)  # mean pooling

w2v_sent_sims = []
tf_sent_sims  = []
pair_labels   = []

for s1, s2, label in sentence_pairs:
    w2v_sent_sims.append(cosine_sim(sentence_w2v_vec(s1), sentence_w2v_vec(s2)))
    tf_sent_sims.append(cosine_sim(sentence_tf_vec(s1),  sentence_tf_vec(s2)))
    pair_labels.append(label)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(pair_labels))
w = 0.35
ax.bar(x - w/2, w2v_sent_sims, w, label="Word2Vec (avg pooling)", color="#E07B54", edgecolor="white")
ax.bar(x + w/2, tf_sent_sims,  w, label="Transformer (mean pool)", color="#5B9BD5", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(pair_labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Cosine similarity", fontsize=11)
ax.set_title("Sentence-level similarity: Word2Vec vs. Transformer", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
for i, (v1, v2) in enumerate(zip(w2v_sent_sims, tf_sent_sims)):
    ax.text(i - w/2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=8)
    ax.text(i + w/2, v2 + 0.02, f"{v2:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}fig5_sentence_similarity.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig5")

print("--- Figure 6: Negative example – negation ---")

negation_pairs = [
    ("patient is healthy",          "patient is not healthy"),
    ("the drug is effective",       "the drug is not effective"),
    ("discharge was successful",    "discharge was not successful"),
    ("lead levels are safe",        "lead levels are not safe"),
    ("cold symptoms are mild",      "cold symptoms are not mild"),
]

w2v_neg_sims = []
tf_neg_sims  = []
neg_labels   = []

for s1, s2 in negation_pairs:
    w2v_neg_sims.append(cosine_sim(sentence_w2v_vec(s1), sentence_w2v_vec(s2)))
    tf_neg_sims.append(cosine_sim(sentence_tf_vec(s1),  sentence_tf_vec(s2)))
    neg_labels.append(s1.replace(" ", "\n", 2))

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(neg_labels))
w = 0.35
ax.bar(x - w/2, w2v_neg_sims, w, label="Word2Vec", color="#E07B54", edgecolor="white")
ax.bar(x + w/2, tf_neg_sims,  w, label="Transformer", color="#5B9BD5", edgecolor="white")
ax.axhline(0.95, color="red", ls="--", lw=1, label="High similarity threshold (0.95)")
ax.set_xticks(x)
ax.set_xticklabels(neg_labels, fontsize=8)
ax.set_ylabel("Cosine similarity", fontsize=11)
ax.set_title("Negative example: similarity between affirmative & negated sentences\n"
             "(ideal: LOW similarity — negation should change meaning)",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
for i, (v1, v2) in enumerate(zip(w2v_neg_sims, tf_neg_sims)):
    ax.text(i - w/2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=8)
    ax.text(i + w/2, v2 + 0.02, f"{v2:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}fig6_negation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig6")

print("--- Figure 7: Most similar words ---")

target_words = ["discharge", "lead", "bank", "cold"]
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Word2Vec: Top-5 most similar words\n(confirms that static embeddings conflate both meanings)",
             fontsize=12, fontweight="bold")

for ax, wd in zip(axes, target_words):
    try:
        similar = w2v.wv.most_similar(wd, topn=5)
    except KeyError:
        similar = []
    words_sim  = [s[0] for s in similar]
    scores_sim = [s[1] for s in similar]
    colors_sim = ["#E07B54" if i == 0 else "#5B9BD5" for i in range(len(words_sim))]
    bars = ax.barh(words_sim[::-1], scores_sim[::-1], color=colors_sim[::-1], edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_title(f'Most similar to\n"{wd}"', fontsize=10, fontweight="bold")
    ax.set_xlabel("Cosine similarity", fontsize=8)
    ax.tick_params(labelsize=9)
    for bar, score in zip(bars, scores_sim[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}fig7_most_similar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig7")

numeric_results = {
    "cross_context_similarity": {
        "word2vec": {k: round(float(v), 4) for k, v in cross_sims_w2v.items()},
        "transformer": {k: round(float(v), 4) for k, v in cross_sims_tf.items()},
    },
    "sentence_similarity": {
        pair_labels[i]: {
            "word2vec": round(float(w2v_sent_sims[i]), 4),
            "transformer": round(float(tf_sent_sims[i]), 4),
        }
        for i in range(len(pair_labels))
    },
    "negation_robustness": {
        f"pair_{i+1}": {
            "word2vec": round(float(w2v_neg_sims[i]), 4),
            "transformer": round(float(tf_neg_sims[i]), 4),
        }
        for i in range(len(negation_pairs))
    },
}

with open(f"{OUT}results.json", "w") as f:
    json.dump(numeric_results, f, indent=2)

print("\n=== All figures and results saved ===")
print(json.dumps(numeric_results, indent=2))
