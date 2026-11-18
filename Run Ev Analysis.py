"""
Problem No. 32: Electric-Vehicle Adoption Across Indian States
Student: Chandra Prakash | Roll No.: 24020570057
Course: Machine Learning, B.Sc. (H) Computer Science, Sem VI (University of Delhi)

Reproduces all regression (OLS, Ridge, Lasso, Decision Tree), clustering (K-Means, PCA),
summary statistics, CSV export, and publication figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# 1. Official MoRTH Vahan4 Dataset as on 14-07-2023 (PIB PRID 1941444)
vahan_rows = [
    (1, "Andaman & Nicobar Island", 160375, 186, 0.12),
    (2, "Andhra Pradesh", 16517516, 66500, 0.40),
    (3, "Arunachal Pradesh", 299371, 25, 0.01),
    (4, "Assam", 5393542, 116605, 2.16),
    (5, "Bihar", 11728184, 155457, 1.33),
    (6, "Chandigarh", 843049, 7628, 0.90),
    (7, "Chhattisgarh", 7431353, 52813, 0.71),
    (8, "Delhi", 8457200, 229305, 2.71),
    (9, "Goa", 1204110, 12139, 1.01),
    (10, "Gujarat", 22799866, 134273, 0.59),
    (11, "Haryana", 12092054, 67812, 0.56),
    (12, "Himachal Pradesh", 2145062, 2362, 0.11),
    (13, "Jammu and Kashmir", 2048212, 10225, 0.50),
    (14, "Jharkhand", 7056955, 35331, 0.50),
    (15, "Karnataka", 29855843, 239948, 0.80),
    (16, "Kerala", 16643512, 94346, 0.57),
    (17, "Ladakh", 43757, 65, 0.15),
    (18, "Madhya Pradesh", 19604968, 92388, 0.47),
    (19, "Maharashtra", 34371551, 296885, 0.86),
    (20, "Manipur", 554096, 1198, 0.22),
    (21, "Meghalaya", 511744, 129, 0.03),
    (22, "Mizoram", 349287, 114, 0.03),
    (23, "Nagaland", 414439, 60, 0.01),
    (24, "Odisha", 10637750, 60097, 0.56),
    (25, "Puducherry", 1329787, 4421, 0.33),
    (26, "Punjab", 13175075, 34162, 0.26),
    (27, "Rajasthan", 18914170, 175595, 0.93),
    (28, "Sikkim", 108442, 20, 0.02),
    (29, "Tamil Nadu", 31643747, 167216, 0.53),
    (30, "Tripura", 711282, 14379, 2.02),
    (31, "UT of DNH &DD", 372133, 345, 0.09),
    (32, "Uttarakhand", 3626246, 48250, 1.33),
    (33, "Uttar Pradesh", 43943230, 556629, 1.27),
    (34, "West Bengal", 15020616, 67111, 0.45),
]

df_vahan = pd.DataFrame(
    vahan_rows,
    columns=[
        "Sl_No",
        "State_UT",
        "Total_Vehicles_Registered",
        "Total_EV_Registered",
        "EV_Share_Pct_Reported",
    ],
)
df_vahan["EV_Share_Pct"] = (
    df_vahan["Total_EV_Registered"] / df_vahan["Total_Vehicles_Registered"] * 100.0
)
df_vahan[
    ["Sl_No", "State_UT", "Total_Vehicles_Registered", "Total_EV_Registered", "EV_Share_Pct_Reported"]
].to_csv("ev_vahan4_states_2023.csv", index=False)

# 2. Supplementary Data: Per Capita NSDP 2021-22 (Current Prices, INR) - PIB PRID 1942055
nsdp_2021_22 = {
    "Andaman & Nicobar Island": np.nan, "Andhra Pradesh": 192587, "Arunachal Pradesh": 215897,
    "Assam": 102965, "Bihar": 49470, "Chandigarh": 349373, "Chhattisgarh": 120704,
    "Delhi": 389529, "Goa": 472070, "Gujarat": 250100, "Haryana": 264835,
    "Himachal Pradesh": 201854, "Jammu and Kashmir": 116619, "Jharkhand": 78660,
    "Karnataka": 265623, "Kerala": 228767, "Ladakh": np.nan, "Madhya Pradesh": 121594,
    "Maharashtra": 215233, "Manipur": 84345, "Meghalaya": 90638, "Mizoram": 188839,
    "Nagaland": 142363, "Odisha": 128873, "Puducherry": 209890, "Punjab": 161888,
    "Rajasthan": 135962, "Sikkim": 472543, "Tamil Nadu": 241131, "Tripura": 140803,
    "UT of DNH &DD": np.nan, "Uttarakhand": 211657, "Uttar Pradesh": 70792, "West Bengal": 124798,
}

# 3. EV Public Charging Stations (EVPCS) - PIB PRID 2086533 (Dec 2024)
evpcs_map = {
    "Andaman & Nicobar Island": 4, "Andhra Pradesh": 601, "Arunachal Pradesh": 41,
    "Assam": 276, "Bihar": 347, "Chandigarh": 13, "Chhattisgarh": 271,
    "Delhi": 1941, "Goa": 137, "Gujarat": 992, "Haryana": 709,
    "Himachal Pradesh": 106, "Jammu and Kashmir": 159, "Jharkhand": 256,
    "Karnataka": 5765, "Kerala": 1212, "Ladakh": 1, "Madhya Pradesh": 903,
    "Maharashtra": 3728, "Manipur": 46, "Meghalaya": 43, "Mizoram": 12,
    "Nagaland": 28, "Odisha": 488, "Puducherry": 41, "Punjab": 593,
    "Rajasthan": 1129, "Sikkim": 5, "Tamil Nadu": 1413, "Tripura": 50,
    "UT of DNH &DD": 6, "Uttarakhand": 177, "Uttar Pradesh": 1989, "West Bengal": 763,
}

# 4. Urbanisation % (Census 2011), 2023 Population Estimate, Area (km2)
urban_pct_map = {
    "Andaman & Nicobar Island": 37.70, "Andhra Pradesh": 29.47, "Arunachal Pradesh": 22.94,
    "Assam": 14.10, "Bihar": 11.29, "Chandigarh": 97.25, "Chhattisgarh": 23.24,
    "Delhi": 97.50, "Goa": 62.17, "Gujarat": 42.60, "Haryana": 34.88,
    "Himachal Pradesh": 10.03, "Jammu and Kashmir": 26.11, "Jharkhand": 24.05,
    "Karnataka": 38.67, "Kerala": 47.70, "Ladakh": 84.00, "Madhya Pradesh": 27.63,
    "Maharashtra": 45.22, "Manipur": 30.21, "Meghalaya": 20.07, "Mizoram": 52.11,
    "Nagaland": 28.86, "Odisha": 16.69, "Puducherry": 68.33, "Punjab": 37.48,
    "Rajasthan": 24.87, "Sikkim": 25.15, "Tamil Nadu": 48.40, "Tripura": 26.17,
    "UT of DNH &DD": 58.43, "Uttarakhand": 30.23, "Uttar Pradesh": 22.27, "West Bengal": 31.87,
}

pop_2023_map = {
    "Andaman & Nicobar Island": 403000, "Andhra Pradesh": 53156000, "Arunachal Pradesh": 1562000,
    "Assam": 35713000, "Bihar": 126756000, "Chandigarh": 1231000, "Chhattisgarh": 30180000,
    "Delhi": 21359000, "Goa": 1575000, "Gujarat": 71507000, "Haryana": 30209000,
    "Himachal Pradesh": 7468000, "Jammu and Kashmir": 13603000, "Jharkhand": 39466000,
    "Karnataka": 67692000, "Kerala": 35776000, "Ladakh": 300000, "Madhya Pradesh": 86579000,
    "Maharashtra": 126710000, "Manipur": 3223000, "Meghalaya": 3349000, "Mizoram": 1238000,
    "Nagaland": 2233000, "Odisha": 46276000, "Puducherry": 1646000, "Punjab": 30730000,
    "Rajasthan": 81025000, "Sikkim": 689000, "Tamil Nadu": 76860000, "Tripura": 4147000,
    "UT of DNH &DD": 1263000, "Uttarakhand": 11637000, "Uttar Pradesh": 235687000, "West Bengal": 99084000,
}

area_km2_map = {
    "Andaman & Nicobar Island": 8249, "Andhra Pradesh": 162970, "Arunachal Pradesh": 83743,
    "Assam": 78438, "Bihar": 94163, "Chandigarh": 114, "Chhattisgarh": 135192,
    "Delhi": 1483, "Goa": 3702, "Gujarat": 196244, "Haryana": 44212,
    "Himachal Pradesh": 55673, "Jammu and Kashmir": 55538, "Jharkhand": 79716,
    "Karnataka": 191791, "Kerala": 38852, "Ladakh": 166698, "Madhya Pradesh": 308252,
    "Maharashtra": 307713, "Manipur": 22327, "Meghalaya": 22429, "Mizoram": 21081,
    "Nagaland": 16579, "Odisha": 155707, "Puducherry": 490, "Punjab": 50362,
    "Rajasthan": 342239, "Sikkim": 7096, "Tamil Nadu": 130060, "Tripura": 10486,
    "UT of DNH &DD": 603, "Uttarakhand": 53483, "Uttar Pradesh": 240928, "West Bengal": 88752,
}

df = df_vahan.copy()
df["NSDP_Per_Capita_INR"] = df["State_UT"].map(nsdp_2021_22)
df["EVPCS"] = df["State_UT"].map(evpcs_map)
df["Urban_Pct"] = df["State_UT"].map(urban_pct_map)
df["Pop_2023"] = df["State_UT"].map(pop_2023_map)
df["Area_km2"] = df["State_UT"].map(area_km2_map)

df["NSDP_Lakh"] = df["NSDP_Per_Capita_INR"] / 1e5
df["Stations_per_100k_Veh"] = df["EVPCS"] / (df["Total_Vehicles_Registered"] / 1e5)
df["Stations_per_1000_km2"] = df["EVPCS"] / (df["Area_km2"] / 1000.0)
df["Veh_per_1000_Pop"] = df["Total_Vehicles_Registered"] / (df["Pop_2023"] / 1000.0)
df["Log_Stations_per_1000_km2"] = np.log1p(df["Stations_per_1000_km2"])
df["Log_Total_Vehicles"] = np.log10(df["Total_Vehicles_Registered"])

# Leakage-free preprocessing & syllabus ML implementations
def preprocess_fit_transform(X_train, X_test):
    med = np.nanmedian(X_train, axis=0)
    X_tr = np.where(np.isnan(X_train), med, X_train)
    X_te = np.where(np.isnan(X_test), med, X_test)
    mu = X_tr.mean(axis=0)
    sd = np.where(X_tr.std(axis=0) == 0, 1.0, X_tr.std(axis=0))
    return (X_tr - mu) / sd, (X_te - mu) / sd

def fit_ridge(X, y, alpha=0.0):
    _, p = X.shape
    y_mean = y.mean()
    w = np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ (y - y_mean))
    return w, y_mean

def fit_lasso(X, y, alpha=0.05, n_iter=2000):
    n, p = X.shape
    y_mean = y.mean()
    yc = y - y_mean
    w = np.zeros(p)
    col_norms = np.sum(X ** 2, axis=0)
    for _ in range(n_iter):
        for j in range(p):
            r_j = yc - X @ w + w[j] * X[:, j]
            rho = X[:, j] @ r_j
            if rho < -n * alpha:
                w[j] = (rho + n * alpha) / col_norms[j]
            elif rho > n * alpha:
                w[j] = (rho - n * alpha) / col_norms[j]
            else:
                w[j] = 0.0
    return w, y_mean

class SimpleDecisionTreeReg:
    def __init__(self, max_depth=3, min_samples_leaf=4):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return float(np.mean(y))
        best_sse, best_feat, best_thr = np.inf, None, None
        for j in range(X.shape[1]):
            vals = np.unique(X[:, j])
            if len(vals) <= 1:
                continue
            for thr in (vals[:-1] + vals[1:]) / 2.0:
                left, right = y[X[:, j] <= thr], y[X[:, j] > thr]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                sse = np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
                if sse < best_sse:
                    best_sse, best_feat, best_thr = sse, j, thr
        if best_feat is None:
            return float(np.mean(y))
        mask = X[:, best_feat] <= best_thr
        return (best_feat, best_thr, self._build(X[mask], y[mask], depth + 1), self._build(X[~mask], y[~mask], depth + 1))

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self

    def _pred_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        f, thr, l, r = node
        return self._pred_one(x, l if x[f] <= thr else r)

    def predict(self, X):
        return np.array([self._pred_one(x, self.tree) for x in X])

features = ["NSDP_Lakh", "Urban_Pct", "Stations_per_100k_Veh", "Log_Stations_per_1000_km2", "Veh_per_1000_Pop", "Log_Total_Vehicles"]
X_raw = df[features].values
y = df["EV_Share_Pct"].values
n = len(y)

rng = np.random.default_rng(42)
folds = np.array_split(rng.permutation(n), 5)
loo_splits = [np.array([i]) for i in range(n)]

def eval_cv(model_fn, split_list):
    preds = np.zeros(n)
    for test_idx in split_list:
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        X_tr, X_te = preprocess_fit_transform(X_raw[train_idx], X_raw[test_idx])
        preds[test_idx] = model_fn(X_tr, y[train_idx], X_te)
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    mae = np.mean(np.abs(y - preds))
    r2 = 1.0 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    return preds, rmse, mae, r2

# Hyper-parameter tuning on 5-fold CV
best_ridge_a = min([0.1, 1.0, 5.0, 10.0, 20.0, 50.0], key=lambda a: eval_cv(lambda xtr, ytr, xte, al=a: xte @ fit_ridge(xtr, ytr, al)[0] + fit_ridge(xtr, ytr, al)[1], folds)[1])
best_lasso_a = min([0.005, 0.01, 0.03, 0.05, 0.1, 0.2], key=lambda a: eval_cv(lambda xtr, ytr, xte, al=a: xte @ fit_lasso(xtr, ytr, al)[0] + fit_lasso(xtr, ytr, al)[1], folds)[1])
best_dt = min([(d, ml) for d in [2, 3, 4] for ml in [3, 4, 5, 6]], key=lambda p: eval_cv(lambda xtr, ytr, xte, dd=p[0], mll=p[1]: SimpleDecisionTreeReg(dd, mll).fit(xtr, ytr).predict(xte), folds)[1])

model_specs = [
    ("Mean Baseline", lambda xtr, ytr, xte: np.full(len(xte), ytr.mean())),
    ("Multiple Linear Reg. (OLS)", lambda xtr, ytr, xte: xte @ fit_ridge(xtr, ytr, 0.0)[0] + fit_ridge(xtr, ytr, 0.0)[1]),
    (f"Ridge (alpha={best_ridge_a})", lambda xtr, ytr, xte: xte @ fit_ridge(xtr, ytr, best_ridge_a)[0] + fit_ridge(xtr, ytr, best_ridge_a)[1]),
    (f"Lasso (alpha={best_lasso_a})", lambda xtr, ytr, xte: xte @ fit_lasso(xtr, ytr, best_lasso_a)[0] + fit_lasso(xtr, ytr, best_lasso_a)[1]),
    (f"Decision Tree {best_dt}", lambda xtr, ytr, xte: SimpleDecisionTreeReg(*best_dt).fit(xtr, ytr).predict(xte)),
]

print("=== REGRESSION RESULTS ===")
for name, fn in model_specs:
    _, r5, m5, r2_5 = eval_cv(fn, folds)
    _, rl, ml, r2_l = eval_cv(fn, loo_splits)
    print(f"{name:28s} | 5F-RMSE={r5:.3f}, 5F-R2={r2_5:.3f} | LOO-RMSE={rl:.3f}, LOO-MAE={ml:.3f}, LOO-R2={r2_l:.3f}")

# K-Means Clustering (k=2..7) + PCA
cluster_features = ["EV_Share_Pct", "NSDP_Lakh", "Urban_Pct", "Stations_per_100k_Veh"]
X_c, _ = preprocess_fit_transform(df[cluster_features].values, df[cluster_features].values)

def kmeans_np(X, k, n_init=50, max_iter=200, seed=42):
    rng_k = np.random.default_rng(seed)
    best_inertia, best_labels, best_centers = np.inf, None, None
    for _ in range(n_init):
        centers = [X[rng_k.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            centers.append(X[rng_k.choice(len(X), p=d2 / d2.sum())])
        centers = np.array(centers)
        for _ in range(max_iter):
            labels = np.argmin(np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2), axis=1)
            new_centers = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j] for j in range(k)])
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        inertia = sum(np.sum((X[labels == j] - centers[j]) ** 2) for j in range(k))
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels, centers
    return best_labels, best_centers, best_inertia

def silhouette_np(X, labels):
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    s_vals = []
    for i in range(len(X)):
        same = labels == labels[i]
        a_i = dists[i, same].sum() / (same.sum() - 1) if same.sum() > 1 else 0.0
        b_i = min(dists[i, labels == c].mean() for c in np.unique(labels) if c != labels[i])
        s_vals.append((b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0)
    return float(np.mean(s_vals))

print("\n=== K-MEANS CLUSTERING (k=2..7) ===")
for k in range(2, 8):
    lbls, _, iner = kmeans_np(X_c, k)
    print(f"k={k}: Inertia={iner:.2f}, Silhouette={silhouette_np(X_c, lbls):.3f}")

lbls3, _, _ = kmeans_np(X_c, 3)
df["Cluster"] = lbls3
print("\n=== CLUSTER ASSIGNMENTS (k=3) ===")
for c in range(3):
    print(f"Cluster {c}: {', '.join(df[df['Cluster'] == c]['State_UT'].tolist())}")