import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
import statsmodels.api as sm

def predict_previous_season(df, model, exclude_year=2025):
    accuracy_arr = []
    roc_arr = []
    for year in range(2004, 2025 + 1):
        df_year = df[df['season'] == year]
        

        # Sanity check
        if len(df_year) == 0 or year == exclude_year:
            continue

        X = df_year[[
            "shots",
            "power_play_goals",
            "faceoff_win_pct",
            "hits",
            "blocked_shots",
            "giveaways",
            "takeaways"
        ]].fillna(0)
        X = sm.add_constant(X, has_constant="add")
        y = df_year["won"]

        pred_prob = model.predict(X)
        pred_class = (pred_prob >= 0.5).astype(int)

        roc = roc_auc_score(y, pred_class)
        acc = accuracy_score(y, pred_class)
        print(f"{year}: Accuracy: {acc:.3f}, ROC_AUC: {roc:.3f}")

        accuracy_arr.append(acc)
        roc_arr.append(roc)
    
    print(
        f"Average accuracy score: {np.mean(accuracy_arr):.3f} "
        f"and Average ROC_AUC: {np.mean(roc_arr):.3f} "
        f"over {len(accuracy_arr)} test seasons"
    )