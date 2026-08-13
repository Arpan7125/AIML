"""
Fix for lab08_2547116.ipynb - Cell 35:

CategoricalNB.feature_log_prob_ in newer sklearn is a list of 2D arrays,
one per feature: feature_log_prob_[feature_idx] has shape (n_classes, n_categories).
The old 3D indexing [class_idx, feature_idx, cat_idx] fails with 'list indices must be
integers or slices, not tuple'.

Fix: Replace all [0, f_idx, cat_idx] and [1, f_idx, cat_idx] indexing patterns
     with the correct [f_idx][0, int(cat_idx)] and [f_idx][1, int(cat_idx)] form.

Also fix the same pattern in Cell 39 (ROC/feature importance cell) which has
the same log_prob indexing.
"""

import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

notebook_path = 'lab08_2547116.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def set_source(cell, new_src):
    lines = new_src.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    cell['outputs'] = [o for o in cell.get('outputs', []) if o.get('output_type') != 'error']
    cell['execution_count'] = None

fixes = []

# ─────────────────────────────────────────────────────────────────────────────
# Fix CategoricalNB feature_log_prob_ indexing in ALL code cells
# Old pattern: nb_model.feature_log_prob_[CLASS, f_idx, cat_idx]
# New pattern: nb_model.feature_log_prob_[f_idx][CLASS, int(cat_idx)]
# ─────────────────────────────────────────────────────────────────────────────
for i, cell in enumerate(cells):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    
    # Pattern: feature_log_prob_[0, f_idx, cat_idx] -> feature_log_prob_[f_idx][0, int(cat_idx)]
    old1 = 'nb_model.feature_log_prob_[0, f_idx, cat_idx]'
    new1 = 'nb_model.feature_log_prob_[f_idx][0, int(cat_idx)]'
    old2 = 'nb_model.feature_log_prob_[1, f_idx, cat_idx]'
    new2 = 'nb_model.feature_log_prob_[f_idx][1, int(cat_idx)]'

    if old1 in src or old2 in src:
        new_src = src.replace(old1, new1).replace(old2, new2)
        set_source(cell, new_src)
        fixes.append(f'Cell {i}: Fixed feature_log_prob_ indexing')

# ─────────────────────────────────────────────────────────────────────────────
# Cell 39 also accesses feature_log_prob_ differently:
#   log_probs = nb_model.feature_log_prob_  # shape: (n_classes, n_features, n_categories)
#   diff = np.abs(log_probs[0, f_idx, :] - log_probs[1, f_idx, :]).mean()
# This is also wrong - fix it
# ─────────────────────────────────────────────────────────────────────────────
cell39 = cells[39]
src39 = ''.join(cell39.get('source', []))

old_block = (
    '    log_probs = nb_model.feature_log_prob_  # shape: (n_classes, n_features, n_categories)\n'
    '    # For each feature, compute the difference in log probs between classes\n'
    '    feat_diff = []\n'
    '    for f_idx, feature in enumerate(feature_cols):\n'
    '        # Average absolute difference across categories\n'
    '        diff = np.abs(log_probs[0, f_idx, :] - log_probs[1, f_idx, :]).mean()\n'
    '        feat_diff.append({\'Feature\': feature, \'Avg LogProb Diff\': diff})'
)
new_block = (
    '    log_probs = nb_model.feature_log_prob_  # list of arrays, one per feature\n'
    '    # For each feature, compute the difference in log probs between classes\n'
    '    feat_diff = []\n'
    '    for f_idx, feature in enumerate(feature_cols):\n'
    '        # Average absolute difference across categories\n'
    '        lp = log_probs[f_idx]  # shape (n_classes, n_categories_for_this_feature)\n'
    '        diff = np.abs(lp[0, :] - lp[1, :]).mean()\n'
    '        feat_diff.append({\'Feature\': feature, \'Avg LogProb Diff\': diff})'
)

if old_block in src39:
    new_src39 = src39.replace(old_block, new_block)
    set_source(cell39, new_src39)
    fixes.append('Cell 39: Fixed feature_log_prob_ list-of-arrays indexing')

# ─────────────────────────────────────────────────────────────────────────────
# Cell 35: Also fix the detailed calculation block at the bottom
# lp_no = nb_model.feature_log_prob_[0, f_idx, cat_idx]  (wrong)
# lp_yes = nb_model.feature_log_prob_[1, f_idx, cat_idx]  (wrong)
# ─────────────────────────────────────────────────────────────────────────────
cell35 = cells[35]
src35 = ''.join(cell35.get('source', []))

old3 = '        lp_no = nb_model.feature_log_prob_[0, f_idx, cat_idx]'
new3 = '        lp_no = nb_model.feature_log_prob_[f_idx][0, int(cat_idx)]'
old4 = '        lp_yes = nb_model.feature_log_prob_[1, f_idx, cat_idx]'
new4 = '        lp_yes = nb_model.feature_log_prob_[f_idx][1, int(cat_idx)]'

if old3 in src35 or old4 in src35:
    new_src35 = src35.replace(old3, new3).replace(old4, new4)
    set_source(cell35, new_src35)
    fixes.append('Cell 35: Fixed bottom feature_log_prob_ indexing')

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
if fixes:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f'Applied {len(fixes)} fix(es):')
    for fix in fixes:
        print(f'  - {fix}')
    print('Notebook saved.')
else:
    print('No patterns matched. Showing relevant snippets:')
    for i in [35, 39]:
        src = ''.join(cells[i].get('source', []))
        if 'feature_log_prob_' in src:
            idx = src.find('feature_log_prob_')
            print(f'Cell {i}: ...{src[max(0,idx-20):idx+100]}...')
