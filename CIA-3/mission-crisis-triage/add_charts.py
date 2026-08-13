import json, copy

NB_PATH = r"f:\backup-2026-08-05\Christ\Sem 4\AIML\CIA-3\mission-crisis-triage\Mission_Crisis_Disaster_Triage.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

def md_cell(source_str, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source_str
    }

def code_cell(source_lines, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

# ─── CHART 1: Priority distribution donut + percentage bar ───────────────────
# Inserted after cell 10 (the priority label code cell, index 10)
chart1_md = md_cell(
    "#### Chart 1 — Priority Class Distribution",
    "chart1_md_a"
)
chart1_code = code_cell(
    [
        "# Chart 1: Priority distribution — donut + horizontal bar\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n",
        "order = ['Critical', 'High', 'Low']\n",
        "colors = ['#d62728', '#ff7f0e', '#2ca02c']\n",
        "counts = df['priority'].value_counts()[order]\n",
        "pcts   = (counts / counts.sum() * 100).round(1)\n",
        "\n",
        "# Donut\n",
        "wedges, texts, autotexts = ax1.pie(\n",
        "    counts, labels=order, colors=colors, autopct='%1.1f%%',\n",
        "    startangle=140, pctdistance=0.78,\n",
        "    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)\n",
        ")\n",
        "for at in autotexts:\n",
        "    at.set_fontsize(11); at.set_fontweight('bold')\n",
        "ax1.set_title('Priority class share (donut)', fontsize=13, pad=14)\n",
        "\n",
        "# Horizontal bar\n",
        "bars = ax2.barh(order, counts.values, color=colors, edgecolor='white', linewidth=0.8)\n",
        "ax2.bar_label(bars, labels=[f'{c:,}  ({p}%)' for c, p in zip(counts, pcts)],\n",
        "               padding=6, fontsize=10)\n",
        "ax2.set_xlabel('Number of messages', fontsize=11)\n",
        "ax2.set_title('Absolute count per priority class', fontsize=13)\n",
        "ax2.set_xlim(0, counts.max() * 1.20)\n",
        "ax2.invert_yaxis()   # Critical on top\n",
        "sns.despine(ax=ax2, left=True)\n",
        "ax2.tick_params(left=False)\n",
        "\n",
        "plt.suptitle('Dataset label overview — 25,984 messages', fontsize=14, y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{FIG_DIR}/chart1_priority_dist.png', dpi=130, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Chart 1 saved.')\n",
    ],
    "chart1_code_a"
)

# ─── CHART 2: Word-length distributions per class (violin) ───────────────────
# Inserted after cell 12 (EDA code cell, index 12) — after eda_overview cell
chart2_md = md_cell(
    "#### Chart 2 — Message Length & Urgency Words by Priority",
    "chart2_md_a"
)
chart2_code = code_cell(
    [
        "# Chart 2: Violin + strip — message length & urgency word count per class\n",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n",
        "order = ['Critical', 'High', 'Low']\n",
        "palette = {'Critical': '#d62728', 'High': '#ff7f0e', 'Low': '#2ca02c'}\n",
        "\n",
        "# Left: message length violin\n",
        "sns.violinplot(x='priority', y='msg_len', data=df, order=order,\n",
        "               palette=palette, inner='quartile', cut=0, ax=axes[0])\n",
        "axes[0].set_title('Message character length by priority', fontsize=12)\n",
        "axes[0].set_xlabel('')\n",
        "axes[0].set_ylabel('Characters', fontsize=11)\n",
        "axes[0].set_ylim(0, df['msg_len'].quantile(0.99))\n",
        "\n",
        "# Right: urgency_word_count strip / count\n",
        "urgency_mean = df.groupby('priority')['urgency_word_count'].mean()[order]\n",
        "bars = axes[1].bar(order,\n",
        "                   urgency_mean.values,\n",
        "                   color=[palette[p] for p in order],\n",
        "                   edgecolor='white', linewidth=0.8)\n",
        "axes[1].bar_label(bars, fmt='%.3f', padding=3, fontsize=10)\n",
        "axes[1].set_title('Mean urgency-keyword count per message by priority', fontsize=12)\n",
        "axes[1].set_xlabel('')\n",
        "axes[1].set_ylabel('Mean urgency words', fontsize=11)\n",
        "sns.despine(ax=axes[1])\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{FIG_DIR}/chart2_length_urgency.png', dpi=130, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Chart 2 saved.')\n",
    ],
    "chart2_code_a"
)

# ─── CHART 3: Genre breakdown stacked bar (detailed) ─────────────────────────
chart3_md = md_cell(
    "#### Chart 3 — Genre × Priority Heatmap",
    "chart3_md_a"
)
chart3_code = code_cell(
    [
        "# Chart 3: Genre x Priority cross-tab heatmap\n",
        "import matplotlib.ticker as mticker\n",
        "cross = pd.crosstab(df['genre'], df['priority'])[['Critical', 'High', 'Low']]\n",
        "cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n",
        "\n",
        "sns.heatmap(cross, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0],\n",
        "            linewidths=0.5, cbar_kws={'label': 'Count'})\n",
        "axes[0].set_title('Genre × Priority — absolute counts', fontsize=12)\n",
        "axes[0].set_xlabel('Priority'); axes[0].set_ylabel('Genre')\n",
        "\n",
        "sns.heatmap(cross_pct, annot=True, fmt='.1f', cmap='Blues', ax=axes[1],\n",
        "            linewidths=0.5, cbar_kws={'label': '% within genre'})\n",
        "axes[1].set_title('Genre × Priority — row-normalised %', fontsize=12)\n",
        "axes[1].set_xlabel('Priority'); axes[1].set_ylabel('')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{FIG_DIR}/chart3_genre_priority_heatmap.png', dpi=130, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Chart 3 saved.')\n",
    ],
    "chart3_code_a"
)

# ─── CHART 4: Metrics comparison grouped bar (after results cell 31) ─────────
chart4_md = md_cell(
    "#### Chart 4 — Full Metrics Comparison Across All Models",
    "chart4_md_a"
)
chart4_code = code_cell(
    [
        "# Chart 4: Grouped bar — all metrics side-by-side\n",
        "metrics = ['Accuracy', 'F1 (macro)', 'F1 (weighted)',\n",
        "           'Precision (macro)', 'Recall (macro)', 'ROC-AUC (ovr, macro)']\n",
        "short_labels = ['Accuracy', 'F1\\n(macro)', 'F1\\n(weighted)',\n",
        "                 'Precision', 'Recall', 'ROC-AUC']\n",
        "model_names = results_df.index.tolist()\n",
        "x = np.arange(len(metrics))\n",
        "width = 0.18\n",
        "model_colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#9467bd']\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(14, 6))\n",
        "for i, (mname, col) in enumerate(zip(model_names, model_colors)):\n",
        "    vals = results_df.loc[mname, metrics].values\n",
        "    rects = ax.bar(x + i * width, vals, width, label=mname,\n",
        "                   color=col, edgecolor='white', linewidth=0.6, alpha=0.88)\n",
        "    ax.bar_label(rects, fmt='%.2f', padding=2, fontsize=7.5, rotation=90)\n",
        "\n",
        "ax.set_xticks(x + width * 1.5)\n",
        "ax.set_xticklabels(short_labels, fontsize=11)\n",
        "ax.set_ylim(0, 1.12)\n",
        "ax.set_ylabel('Score', fontsize=12)\n",
        "ax.set_title('All evaluation metrics — Baseline → Bagging → Boosting → Stacking',\n",
        "             fontsize=13)\n",
        "ax.legend(loc='lower right', fontsize=9, title='Model')\n",
        "ax.axhline(0.70, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)\n",
        "ax.text(len(metrics) - 0.1, 0.705, '0.70 target', color='grey', fontsize=8, ha='right')\n",
        "sns.despine(ax=ax)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{FIG_DIR}/chart4_metrics_comparison.png', dpi=130, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Chart 4 saved.')\n",
    ],
    "chart4_code_a"
)

# ─── CHART 5: Critical-class recall bar ──────────────────────────────────────
chart5_md = md_cell(
    "#### Chart 5 — Critical-Class Recall per Model",
    "chart5_md_a"
)
chart5_code = code_cell(
    [
        "# Chart 5: Critical-class recall — the mission-critical metric\n",
        "critical_idx_c = list(le.classes_).index('Critical')\n",
        "crit_recall = {}\n",
        "for mname, model in models.items():\n",
        "    pred = model.predict(X_test_mat)\n",
        "    crit_recall[mname] = recall_score(y_test, pred, labels=[critical_idx_c], average='macro')\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(9, 5))\n",
        "mnames = list(crit_recall.keys())\n",
        "vals   = list(crit_recall.values())\n",
        "bar_cols = ['#8c564b', '#1f77b4', '#ff7f0e', '#9467bd']\n",
        "bars = ax.bar(mnames, vals, color=bar_cols, edgecolor='white', linewidth=0.8)\n",
        "ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=11, fontweight='bold')\n",
        "ax.set_ylim(0, 0.80)\n",
        "ax.set_ylabel('Recall (Critical class)', fontsize=12)\n",
        "ax.set_title('How well does each model catch life-threatening messages?\\n'\n",
        "             '(Critical-class recall — higher = fewer missed emergencies)', fontsize=12)\n",
        "ax.axhline(0.61, color='#d62728', linewidth=1.2, linestyle='--')\n",
        "ax.text(len(mnames) - 0.5, 0.615, 'XGB best = 0.610', color='#d62728', fontsize=9, ha='right')\n",
        "plt.xticks(rotation=15, ha='right', fontsize=10)\n",
        "sns.despine(ax=ax)\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{FIG_DIR}/chart5_critical_recall.png', dpi=130, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Chart 5 saved.')\n",
    ],
    "chart5_code_a"
)

# ─── CHART 6: Per-class ROC curves for best model (Stacking) ─────────────────
chart6_md = md_cell(
    "#### Chart 6 — Per-class ROC Curves (Best Model: Stacking)",
    "chart6_md_a"
)
chart6_code = code_cell(
    [
        "# Chart 6: Per-class OvR ROC curves for the stacking ensemble\n",
        "from sklearn.metrics import roc_curve, auc\n",
        "\n",
        "y_test_bin_roc = label_binarize(y_test, classes=[0, 1, 2])\n",
        "best_proba = best_model.predict_proba(X_test_mat)\n",
        "class_colors = ['#d62728', '#ff7f0e', '#2ca02c']\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8, 6))\n",
        "for c_i, (cls_name, col) in enumerate(zip(le.classes_, class_colors)):\n",
        "    fpr, tpr, _ = roc_curve(y_test_bin_roc[:, c_i], best_proba[:, c_i])\n",
        "    roc_auc = auc(fpr, tpr)\n",
        "    ax.plot(fpr, tpr, lw=2, color=col,\n",
        "            label=f'{cls_name}  (AUC = {roc_auc:.3f})')\n",
        "\n",
        "ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)\n",
        "ax.fill_between([0, 1], [0, 1], alpha=0.03, color='grey')\n",
        "ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])\n",
        "ax.set_xlabel('False Positive Rate', fontsize=12)\n",
        "ax.set_ylabel('True Positive Rate', fontsize=12)\n",
        "ax.set_title('One-vs-Rest ROC curves — Stacking (RF+XGB+LogReg)', fontsize=13)\n",
        "ax.legend(loc='lower right', fontsize=11)\n",
        "sns.despine(ax=ax)\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{FIG_DIR}/chart6_roc_curves.png', dpi=130, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Chart 6 saved.')\n",
    ],
    "chart6_code_a"
)

# ─── CHART 7: Feature importance from Random Forest (top 20 words) ───────────
chart7_md = md_cell(
    "#### Chart 7 — Top-20 TF-IDF Feature Importances (Random Forest)",
    "chart7_md_a"
)
chart7_code = code_cell(
    [
        "# Chart 7: RF feature importances — top 20\n",
        "importances = rf_best.feature_importances_\n",
        "feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)\n",
        "top20 = feat_imp.head(20)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(10, 6))\n",
        "bars = ax.barh(top20.index[::-1], top20.values[::-1],\n",
        "               color='#1f77b4', edgecolor='white', linewidth=0.6)\n",
        "ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)\n",
        "ax.set_xlabel('Mean Decrease in Impurity (Gini)', fontsize=11)\n",
        "ax.set_title('Top-20 feature importances — Random Forest\\n'\n",
        "             '(TF-IDF tokens + engineered features)', fontsize=12)\n",
        "sns.despine(ax=ax)\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{FIG_DIR}/chart7_rf_feature_importance.png', dpi=130, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Chart 7 saved.')\n",
    ],
    "chart7_code_a"
)

# ─── CHART 8: Precision-Recall per class (best model) ────────────────────────
chart8_md = md_cell(
    "#### Chart 8 — Per-class Precision-Recall Curves (Best Model: Stacking)",
    "chart8_md_a"
)
chart8_code = code_cell(
    [
        "# Chart 8: Precision-Recall curves for best model\n",
        "from sklearn.metrics import precision_recall_curve, average_precision_score\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8, 6))\n",
        "for c_i, (cls_name, col) in enumerate(zip(le.classes_, class_colors)):\n",
        "    prec_c, rec_c, _ = precision_recall_curve(y_test_bin_roc[:, c_i], best_proba[:, c_i])\n",
        "    ap = average_precision_score(y_test_bin_roc[:, c_i], best_proba[:, c_i])\n",
        "    ax.plot(rec_c, prec_c, lw=2, color=col, label=f'{cls_name}  (AP = {ap:.3f})')\n",
        "\n",
        "# Baseline\n",
        "class_freqs = np.bincount(y_test) / len(y_test)\n",
        "for c_i, col in enumerate(class_colors):\n",
        "    ax.axhline(class_freqs[c_i], color=col, linewidth=0.8, linestyle=':', alpha=0.5)\n",
        "\n",
        "ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])\n",
        "ax.set_xlabel('Recall', fontsize=12)\n",
        "ax.set_ylabel('Precision', fontsize=12)\n",
        "ax.set_title('Precision-Recall curves — Stacking (RF+XGB+LogReg)\\n'\n",
        "             '(dotted lines = no-skill baseline)', fontsize=12)\n",
        "ax.legend(loc='upper right', fontsize=11)\n",
        "sns.despine(ax=ax)\n",
        "plt.tight_layout()\n",
        "plt.savefig(f'{FIG_DIR}/chart8_precision_recall.png', dpi=130, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Chart 8 saved.')\n",
    ],
    "chart8_code_a"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Locate insertion points and splice in cells
# ═══════════════════════════════════════════════════════════════════════════════
cells = nb["cells"]

def find_cell(cells, fragment, cell_type=None):
    """Return index of first cell whose source contains `fragment`."""
    for i, c in enumerate(cells):
        if cell_type and c["cell_type"] != cell_type:
            continue
        src = "".join(c.get("source", ""))
        if fragment in src:
            return i
    return None

# 1) After cell 10 (priority label code) → insert Chart 1
idx = find_cell(cells, "assign_priority", "code")
assert idx is not None, "priority code cell not found"
cells.insert(idx + 1, chart1_code)
cells.insert(idx + 1, chart1_md)

# 2) After EDA code cell (eda_overview) → insert Charts 2 & 3
idx = find_cell(cells, "eda_overview.png", "code")
assert idx is not None, "EDA code cell not found"
cells.insert(idx + 1, chart3_code)
cells.insert(idx + 1, chart3_md)
cells.insert(idx + 1, chart2_code)
cells.insert(idx + 1, chart2_md)

# 3) After results cell → insert Charts 4 & 5
idx = find_cell(cells, "results_df.to_csv", "code")
assert idx is not None, "results code cell not found"
cells.insert(idx + 1, chart5_code)
cells.insert(idx + 1, chart5_md)
cells.insert(idx + 1, chart4_code)
cells.insert(idx + 1, chart4_md)

# 4) After confusion matrix cell → insert Charts 6, 7, 8
idx = find_cell(cells, "confusion_matrices.png", "code")
assert idx is not None, "confusion matrices code cell not found"
cells.insert(idx + 1, chart8_code)
cells.insert(idx + 1, chart8_md)
cells.insert(idx + 1, chart7_code)
cells.insert(idx + 1, chart7_md)
cells.insert(idx + 1, chart6_code)
cells.insert(idx + 1, chart6_md)

nb["cells"] = cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done — notebook now has {len(nb['cells'])} cells.")
print("Charts added:")
print("  Chart 1  — Priority class distribution (donut + horizontal bar)")
print("  Chart 2  — Message length & urgency words by priority (violin + bar)")
print("  Chart 3  — Genre × Priority heatmap (counts + % normalised)")
print("  Chart 4  — Full metrics comparison grouped bar chart")
print("  Chart 5  — Critical-class recall per model")
print("  Chart 6  — Per-class OvR ROC curves (best model: Stacking)")
print("  Chart 7  — Top-20 RF feature importances")
print("  Chart 8  — Per-class Precision-Recall curves (best model)")
