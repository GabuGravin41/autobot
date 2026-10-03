import pandas as pd
from IPython.display import display, HTML

experiment_data = [
    {
        "Category": 1, 
        "Notebook": "S6E9 Single XGB CV: 0.94583", 
        "Author": "Дворкин Евгений Владимирович",
        "CV": 0.94583, 
        "LB": 0.94590, 
        "Folds": 10, 
        "Link": "https://www.kaggle.com/code/evgendvorkin/s6e9-single-xgb-cv-0-94583"
    },
    {
        "Category": 1, 
        "Notebook": "S6E9 LightGBM", 
        "Author": "cstdy",
        "CV": 0.945831646, 
        "LB": 0.94638, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/kirill0212/s6e9-lightgbm"
    },

    {
        "Category": 1, 
        "Notebook": "Pure LGBM Notebook CV 0.94607 LB 0.94638", 
        "Author": "Naji Ama",
        "CV": 0.94607, 
        "LB": 0.94638, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/najiama/pure-lgbm-Notebook-cv-0-94607-lb-0-94638"
    },    

    {
        "Category": 1, 
        "Notebook": "PS|S6|E9: RealMLP · PyTorch", 
        "Author": "Vladimir Demidov",
        "CV": 0.94601, 
        "LB": 0.94621, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/yekenot/ps-s6-e9-realmlp-pytorch"
    },   
    {
        "Category": 3, 
        "Notebook": "S6E9 Electric Vehicle OOF", 
        "Author": "Naji Ama",
        "CV": 0.94618, 
        "LB": 0.94633, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/najiama/s6e9-electric-vehicle-oof-cv-0-94618-lb-0-94633"
    },   
    {
        "Category": 2, 
        "Notebook": "EVehicle | Stacked LGBM/CatB/XGB/HGBC | Baseline", 
        "Author": "Ákos Pintér",
        "CV": 0.945990815368376, 
        "LB": 0.94609, 
        "Folds": 7, 
        "Link": "https://www.kaggle.com/code/kospintr/evehicle-stacked-lgbm-catb-xgb-hgbc-baseline"
    },   

    {
        "Category": 1, 
        "Notebook": "Signal that Matters (EDA + HistGBM: LB-0.94637) 😼", 
        "Author": "Aryan Kaisth",
        "CV": 0.94603, 
        "LB": 0.94637, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/aryankaisth/signal-that-matters-eda-histgbm-lb-0-94637"
    },      

    {   "Category": 2, 
        "Notebook": "EV Quantum Forge S6E9 XGBoost", 
        "Author": "Krizsó Gergely",
        "CV": 0.945867, 
        "LB": 0.94611, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/lucifer19/ev-quantum-forge-s6e9-xgboost?scriptVersionId=347918300"},  

    {   "Category": 1, 
        "Notebook": "XGBoost | Triple-TE & Dynamic Pruning (LB 0.94639)", 
        "Author": "Naji Ama",
        "CV": 0.94624, 
        "LB": 0.94639, 
        "Folds": 10, 
        "Link": "https://www.kaggle.com/code/najiama/xgboost-triple-te-dynamic-pruning-lb-0-94639"},  

    {   "Category": 2, 
        "Notebook": "Predicting Electric Vehicle Interest", 
        "Author": "Jocelyn Dumlao",
        "CV": 0.945510822, 
        "LB": 0.94569, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/jocelyndumlao/predicting-electric-vehicle-interest"},  

    {   "Category": 3, 
        "Notebook": "S6E9 Four Feature Views One Ensemble OOF", 
        "Author": "인공저능연구소",
        "CV": 0.946339, 
        "LB": 0.94639, 
        "Folds": 10, 
        "Link": "https://www.kaggle.com/datasets/megayak/s6e9-four-feature-views-one-ensemble-oof"},  

    {   "Category": 3, 
        "Notebook": "🔥 S6E9 | 6-Notebook Meta-Stacking Engine", 
        "Author": "Subodh Deogade",
        "CV": 0.946349, 
        "LB": 0.94644, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/sometimessubodh/s6e9-6-Notebook-meta-stacking-engine?scriptVersionId=350001691"},  

    {   "Category": 1, 
        "Notebook": "simple 21 feature lfbm 0.9456+", 
        "Author": "Sergey Qt2024",
        "CV": 0.0, 
        "LB": 0.94561, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/sergeyqt2024/simple-21-feature-lfbm-0-9456"},  

    {   "Category": 2, 
        "Notebook": "S6E9|CV-0.946072|lb-0.94614", 
        "Author": "Tamerlan Omralinov",
        "CV": 0.946069, 
        "LB": 0.94612, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/tamerlanomralinov/s6e9-cv-0-946072-lb-0-94614"},  

    {   "Category": 1, 
        "Notebook": "Simple TabM Baseline S6E9", 
        "Author": "Yusuke Hayashi",
        "CV": 0.943912, 
        "LB": 0.94423, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/yhay81/simple-tabm-baseline-s6e9?scriptVersionId=347003029"},  

    {   "Category": 1, 
        "Notebook": "S6E9: Deep Neural Network (Entity Embeddings)", 
        "Author": "Aman Atar",
        "CV": 0.944086, 
        "LB": 0.94473, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/amanatar/s6e9-deep-neural-network-entity-embeddings"},  

    {   "Category": 1, 
        "Notebook": "S6E9 Competition-Only LGBM | CV 0.94610/LB 0.94636", 
        "Author": "Mizushima Toshihiko",
        "CV": 0.946099, 
        "LB": 0.94636, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/mizushimatoshihiko/s6e9-competition-only-lgbm-cv-0-94610-lb-0-94636"},  

    {   "Category": 1, 
        "Notebook": "S6E9 | The best DL Notebook + Insights", 
        "Author": "Tamerlan Omralinov",
        "CV": 0.9418773934953342, 
        "LB": 0.94271, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/tamerlanomralinov/s6e9-the-best-dl-Notebook-insights"},  

    {   "Category": 2, 
        "Notebook": "Fable 5.1 - XGB Starter", 
        "Author": "Chris Deotte",
        "CV": 0.94207, 
        "LB": 0.94212, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/cdeotte/fable-5-1-xgb-starter"},  

    {   "Category": 1, 
        "Notebook": "S6E9: GPU Random Forest (cuML)", 
        "Author": "Aman Atar",
        "CV": 0.93803, 
        "LB": 0.93827, 
        "Folds": 10, 
        "Link": "https://www.kaggle.com/code/amanatar/s6e9-gpu-random-forest-cuml"},  

    {   "Category": 1, 
        "Notebook": "S6E9: Tabular ResNet (PyTorch)", 
        "Author": "Aman Atar",
        "CV": 0.93797, 
        "LB": 0.93786, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/amanatar/s6e9-tabular-resnet-pytorch"},      

    {   "Category": 1, 
        "Notebook": "S6E9: FT-Transformer (PyTorch)", 
        "Author": "Aman Atar",
        "CV": 0.93835, 
        "LB": 0.93829, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/amanatar/s6e9-ft-transformer-pytorch"},      

    {   "Category": 1, 
        "Notebook": "CatBoost | Triple-TE & Dynamic Pruning LB 0.94624", 
        "Author": "Naji Ama",
        "CV": 0.94621, 
        "LB": 0.94624, 
        "Folds": 10, 
        "Link": "https://www.kaggle.com/code/najiama/catboost-triple-te-dynamic-pruning-lb-0-94624"},      

    {   "Category": 2, 
        "Notebook": "EV Purchase Prediction Unified Pipeline", 
        "Author": "DYLAN GUNAWAN13331",
        "CV": 0.94540, 
        "LB": 0.94540, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/dylangunawan13331/ev-purchase-prediction-unified-pipeline"},      

    {   "Category": 1, 
        "Notebook": "S6E9: PyTorch TabNet", 
        "Author": "Aman Atar",
        "CV": 0.93779, 
        "LB": 0.93789, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/amanatar/s6e9-pytorch-tabnet"},      

    {   "Category": 1, 
        "Notebook": "S6E9 Leak-Free Target + Frequency Encoding | LGBM", 
        "Author": "Mizushima Toshihiko",
        "CV": 0.945101598, 
        "LB": 0.94560, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/mizushimatoshihiko/s6e9-leak-free-target-frequency-encoding-lgbm"},      

    {   "Category": 1, 
        "Notebook": "BBG007_NN", 
        "Author": "Barbagrande007",
        "CV": 0.9382, 
        "LB": 0.93797, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/barbagrande007/bbg007-nn"},      

     

    {   "Category": 1, 
        "Notebook": "EV-Prophet: A PyTorch Tabular Ensemble", 
        "Author": "Yashpalsinh Thakor",
        "CV": 0.938374, 
        "LB": 0.93787, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/yashpalthakor/ev-prophet-a-pytorch-tabular-ensemble"},      

    {   "Category": 1, 
        "Notebook": "🏆 S6E9 EV Purchase Prediction ", 
        "Author": "Shair Khan",
        "CV": 0.94187, 
        "LB": 0.94156, 
        "Folds": 5, 
        "Link": "https://www.kaggle.com/code/datascikhan/s6e9-ev-purchase-prediction"},      

        {   "Category": 1, 
        "Notebook": "S6E9 🔋🚗| XGBoost + Window Encodings | ⚡0.946 CV", 
        "Author": "BlamerX",
        "CV": 0.946208, 
        "LB": 0.94646, 
        "Folds": 10, 
        "Link": "https://www.kaggle.com/code/blamerx/s6e9-xgboost-window-encodings-0-946-cv"},   

        {   "Category": 1, 
        "Notebook": "KPS6E09 XGB Sample", 
        "Author": "Paul Bryan Elefante",
        "CV": 0.946308910839, 
        "LB": 0.94638, 
        "Folds": 10, 
        "Link": "https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample"},

        {   "Category": 2, 
        "Notebook": "EV Quantum Forge Fusion II Partition Bagged S6E9", 
        "Author": "Agent Trainer",
        "CV": 0.946398, 
        "LB": 0.94645, 
        "Folds":5 , 
        "Link": "https://www.kaggle.com/code/lucifer19/ev-quantum-forge-fusion-ii-partition-bagged-s6e9"},
    
        #     {   "Category": 1, 
        # "Notebook": "", 
        # "Author": "",
        # "CV": 0., 
        # "LB": 0., 
        # "Folds": , 
        # "Link": ""},

    
    
]

# 2. Process Data
df = pd.DataFrame(experiment_data)
df['LB-CV Gap'] = df['LB'] - df['CV']

def make_button(url):
    return f'<a target="_blank" href="{url}" style="color: white; background-color: #20beff; padding: 4px 12px; text-decoration: none; border-radius: 4px; font-weight: bold;">View</a>'

# 3. Define the Three Categories
categories = {
    1: ("🥇 Category 1: Pure Single Models", "Models using only ONE algorithm (XGB, LGBM, Cat, NN, etc.) trained from scratch."),
    2: ("🧬 Category 2: End-to-End Ensembles", "Multiple models trained from scratch and blended within the exact same script."),
    3: ("🗃️ Category 3: OOF & Test Datasets", "Out-Of-Fold and Test predictions for meta-modeling (No Source files).")
}


for cat_id, (cat_title, cat_desc) in categories.items():
    # Print Headers
    display(HTML(f"<h3>{cat_title}</h3><p><i>{cat_desc}</i></p>"))
    
    # Filter for the current category
    cat_df = df[df['Category'] == cat_id].drop(columns=['Category'])
    
    if len(cat_df) == 0:
        display(HTML("<p style='color:gray;'><i>No entries yet. Be the first to share in the comments!</i></p><br><br>"))
        continue
        
    # Sort strictly by LB (Descending)
    cat_df = cat_df.sort_values(by='LB', ascending=False).reset_index(drop=True)
    
    # Rearrange columns for a clean look
    cat_df = cat_df[['Notebook', 'Author', 'CV', 'LB', 'LB-CV Gap', 'Folds', 'Link']]
    
    # Apply pure Pandas Styling (No Javascript needed!)
    styled_df = (cat_df.style
                 .format({
                     'CV': '{:.5f}', 
                     'LB': '{:.5f}', 
                     'LB-CV Gap': '{:+.5f}',
                     'Link': make_button
                 })
                 # .background_gradient(subset=['LB', 'CV'], cmap='Blues') # Adds a nice heat-map color to scores
                 .set_properties(**{'text-align': 'center'})
                 .set_table_styles([dict(selector='th', props=[('text-align', 'center'), ('background-color', '#f2f2f2')])])
                 .hide(axis="index")) # Hides the 0, 1, 2 row numbers for a cleaner look
    
    # Render the table
    display(styled_df)
    display(HTML("<br><br>"))



import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 2. Strip emojis/symbols to avoid font warnings
emoji_pattern = re.compile(r"[\U00010000-\U0010FFFF]|[\u2600-\u27BF]")


def clean_label(text, max_len=30):
    cleaned = emoji_pattern.sub("", text).strip()
    return f"{cleaned[:max_len]}..." if len(cleaned) > max_len else cleaned


categories = {
    1: "Category 1: Pure Single Models",
    2: "Category 2: End-to-End Ensembles",
    3: "Category 3: OOF & Test Datasets",
}

# 3. Vertical Bar Plots for each category
for cat_id, cat_title in categories.items():
    cat_df = df[df["Category"] == cat_id].copy()
    if cat_df.empty:
        continue

    # Sort descending by LB (left-to-right on x-axis)
    cat_df = cat_df.sort_values(by="LB", ascending=False).reset_index(drop=True)

    fig_width = max(4, len(cat_df))
    fig, (ax_lb, ax_gap) = plt.subplots(
        2, 1, figsize=(fig_width, 8), sharex=True
    )
    fig.suptitle(cat_title, fontsize=14, fontweight="bold", y=0.98)

    x_pos = np.arange(len(cat_df))
    x_labels = [clean_label(name) for name in cat_df["Notebook"]]

    # --- Top Plot: LB Score ---
    ax_lb.bar(
        x_pos, cat_df["LB"], color="#20beff", edgecolor="#0094d4", width=0.65
    )
    ax_lb.set_ylabel("LB Score", fontweight="bold")
    ax_lb.set_title("Leaderboard (LB)", fontweight="bold", fontsize=11)
    ax_lb.grid(axis="y", linestyle="--", alpha=0.5)

    lb_min, lb_max = cat_df["LB"].min(), cat_df["LB"].max()
    ax_lb.set_ylim(lb_min - 0.001, lb_max + 0.0005)

    # --- Bottom Plot: LB - CV Gap (scaled by 10^5) ---
    gap_scaled = cat_df["LB-CV Gap"] * 1e5
    valid_gaps = gap_scaled[cat_df["CV"] > 0]
    gap_colors = ["#2ecc71" if g >= 0 else "#e74c3c" for g in gap_scaled]

    ax_gap.bar(x_pos, gap_scaled, color=gap_colors, width=0.65, alpha=0.85)
    ax_gap.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax_gap.set_ylabel(r"LB − CV Gap ($\times 10^5$)", fontweight="bold")
    ax_gap.set_title(
        r"LB − CV Gap ($\times 10^5$)", fontweight="bold", fontsize=11
    )
    ax_gap.grid(axis="y", linestyle="--", alpha=0.5)

    if len(valid_gaps) > 0:
        max_bound = max(abs(valid_gaps.min()), abs(valid_gaps.max())) * 1.3
        ax_gap.set_ylim(-max_bound, max_bound)

    # Shared X-axis ticks (Notebook Names)
    ax_gap.set_xticks(x_pos)
    ax_gap.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)

    plt.tight_layout()
    plt.show()

