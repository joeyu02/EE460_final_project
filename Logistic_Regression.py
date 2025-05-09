# EE 460 Final Project
# Group 1
# Heart Disease Predictor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, SMOTENC 
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, recall_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns 
from collections import Counter
input_file = './project/heart_disease_uci.csv'


def train_val_curve(model, X_train, y_train, X_val, y_val, fracs):
    train_accs, val_accs = [], []
    for frac in fracs:
        n = max(1, int(len(X_train) * frac))
        model.fit(X_train[:n], y_train[:n])
        train_accs.append(model.score(X_train[:n], y_train[:n]))
        val_accs.append(model.score(X_val, y_val))
    return train_accs, val_accs





def read_data(data_location):
    try:
        df = pd.read_csv(data_location, header=None, na_values='?')
    except FileNotFoundError:
        print(f"Error 1")
        return None, None, 0, 0
    
    data_np = df.to_numpy()
    original_rows, original_cols = data_np.shape

    if original_rows <= 1: 
        print("Error 2")
        return None, None, 0, 0

    try:
        class_values = data_np[1:, 15]
        feature_data = data_np[1:, 0:15]
    except IndexError:
        print("Error 3")
        return None, None, 0, 0
        
    return feature_data, class_values, original_rows, original_cols

def average_replace(current_data_arr, column_idx):

    col_numeric = pd.to_numeric(current_data_arr[:, column_idx], errors='coerce').astype(float)
    
    if np.all(np.isnan(col_numeric)):
        mean_val = 0.0 
    else:
        mean_val = np.nanmean(col_numeric) 
    nan_mask = np.isnan(col_numeric)
    current_data_arr[nan_mask, column_idx] = mean_val

    current_data_arr[:, column_idx] = current_data_arr[:, column_idx].astype(float)
    return current_data_arr

data_from_file, class_from_file, _, _ = read_data(input_file)

if data_from_file is None:
    print("Error 4")
    exit()
try:
    class_from_file = class_from_file.astype(int)
except ValueError:
    print("Error 5")
    class_coerced = pd.to_numeric(class_from_file, errors='coerce')
    problematic_indices = np.where(np.isnan(class_coerced))[0]
    if len(problematic_indices) > 0:
        print(f"Error 6")
    exit()

X_processed = data_from_file.copy()

X_processed = X_processed[:, 0:12]


dataset_map = {
    'Cleveland'    : 0,
    'Hungary'      : 1,
    'Switzerland'  : 2,
    'VA Long Beach': 3
}

X_processed[:, 3] = np.vectorize(dataset_map.get)(X_processed[:, 3])

X_processed[:, 2] = np.where(X_processed[:, 2] == 'Male', 1, np.where(X_processed[:, 2] == 'Female', 0, X_processed[:, 2]))
X_processed[:, 7] = np.where(X_processed[:, 7] == 'TRUE', 1, np.where(X_processed[:, 7] == 'FALSE', 0, X_processed[:, 7]))
X_processed[:, 10] = np.where(X_processed[:, 10] == 'TRUE', 1, np.where(X_processed[:, 10] == 'FALSE', 0, X_processed[:, 10]))

for col_idx_binary in [2, 7, 10]:
    X_processed[:, col_idx_binary] = pd.to_numeric(X_processed[:, col_idx_binary], errors='coerce')
numeric_cols_to_impute = [1, 5, 6, 9, 11, 2, 7, 10] 
for ci in numeric_cols_to_impute:
    X_processed = average_replace(X_processed, ci)

X_processed[:, 8] = X_processed[:, 8].astype(object) 
X_processed[pd.isna(X_processed[:, 8]), 8] = "normal"

ohe_categorical_indices = [4, 8]
categorical_cols_for_ohe = X_processed[:, ohe_categorical_indices]

ohe_encoder = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')
one_hot_encoded_features = ohe_encoder.fit_transform(categorical_cols_for_ohe)

ohe_input_feature_names = ['cp', 'restecg'] 
ohe_generated_names = list(ohe_encoder.get_feature_names_out(ohe_input_feature_names))
X_processed = np.delete(X_processed, ohe_categorical_indices, axis=1)
X_processed = np.hstack((X_processed.astype(float), one_hot_encoded_features))

cols_to_normalize_indices = [1, 4, 5, 7, 9] 
features_to_scale = X_processed[:, cols_to_normalize_indices]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_to_scale)


for i, col_idx in enumerate(cols_to_normalize_indices):
    X_processed[:, col_idx] = scaled_features[:, i]

X = X_processed.astype(float)
y = class_from_file 

base_numeric_and_binary_feature_names = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak']
feature_names = base_numeric_and_binary_feature_names + ohe_generated_names

num_base_features = len(base_numeric_and_binary_feature_names) 
smotenc_categorical_features = [1, 4, 6] + list(range(num_base_features, X.shape[1]))
labels_for_smote, counts_for_smote = np.unique(y, return_counts=True)
min_count_smote = 0
if len(counts_for_smote) > 0 :
    min_count_smote = int(counts_for_smote.min())

k_neigh_smote = 1 
if min_count_smote > 1: 
    k_neigh_smote = min(5, min_count_smote - 1)
    if k_neigh_smote < 1 : k_neigh_smote = 1 

if min_count_smote <= k_neigh_smote and min_count_smote > 0: 
    k_neigh_smote = max(1, min_count_smote - 1)


if min_count_smote == 0: 
    print("Error 7")
    X_res, y_res = X.copy(), y.copy()
elif not smotenc_categorical_features or X.shape[1] == len(smotenc_categorical_features): # Check if all features are categorical (SMOTE) or if list is empty
    smote_sampler = SMOTE(random_state=42, k_neighbors=k_neigh_smote)
    X_res, y_res = smote_sampler.fit_resample(X, y)
else:
    try:
        smote_nc_sampler = SMOTENC(categorical_features=smotenc_categorical_features, random_state=42, k_neighbors=k_neigh_smote)
        X_res, y_res = smote_nc_sampler.fit_resample(X, y)
    except ValueError as e:
        print(f"Error 8")
        smote_sampler = SMOTE(random_state=42, k_neighbors=k_neigh_smote)
        X_res, y_res = smote_sampler.fit_resample(X, y)

def split_data(Xd, yd):
    unique_classes_split, counts_split = np.unique(yd, return_counts=True)
    strat_tv = None
    if len(unique_classes_split) > 1 and np.all(counts_split >= 1): 
        strat_tv = yd
    
    Xtv, Xt, ytv, yt = train_test_split(
        Xd, yd, test_size=0.1, random_state=42, stratify=strat_tv
    )

    unique_classes_val, counts_val = np.unique(ytv, return_counts=True)
    strat_val = None
    if len(unique_classes_val) > 1 and np.all(counts_val >= 2): 
        strat_val = ytv

    Xtr, Xv, ytr, yv = train_test_split(
        Xtv, ytv, test_size=0.11, random_state=42, stratify=strat_val 
    )
    return Xtr, ytr, Xv, yv, Xt, yt

print("\nSplitting original")
Xtr, ytr, Xv, yv, Xt, yt = split_data(X, y)
print("Orig shapes - Train:", Xtr.shape, "Validation:", Xv.shape, "Test:", Xt.shape)





clf_unbal = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
clf_unbal.fit(Xtr, ytr)

clf_bal = LogisticRegression(solver='liblinear', max_iter=1000,
                             class_weight='balanced', random_state=42)
clf_bal.fit(Xtr, ytr)
true_counts = Counter(yt)
classes     = sorted(true_counts.keys())
true_freqs  = [true_counts[c] for c in classes]
x           = np.arange(len(classes))
width       = 0.35
preds_ub   = clf_unbal.predict(Xt)
counts_ub  = Counter(preds_ub)
freqs_ub   = [counts_ub.get(c, 0) for c in classes]
plt.figure(figsize=(6,4))
plt.bar(x - width/2, true_freqs, width, label='True')
plt.bar(x + width/2, freqs_ub,   width, label='Predicted (unbalanced)')
plt.xticks(x, classes)
plt.xlabel("Class label")
plt.ylabel("Count")
plt.title("Original Data: No Class Balancing")
plt.legend()
plt.tight_layout()
plt.show(block=True)
preds_b    = clf_bal.predict(Xt)
counts_b   = Counter(preds_b)
freqs_b    = [counts_b.get(c, 0) for c in classes]
plt.figure(figsize=(6,4))
plt.bar(x - width/2, true_freqs, width, label='True')
plt.bar(x + width/2, freqs_b,   width, label='Predicted (balanced)')
plt.xticks(x, classes)
plt.xlabel("Class label")
plt.ylabel("Count")
plt.title("Original Data: With Class Balancing")
plt.legend()
plt.tight_layout()
plt.show(block=True)


print("\nSplitting upsampled")
UXtr, Uytr, UXv, Uyv, UXt, Uyt = split_data(X_res, y_res)
print("Up shapes - Train:", UXtr.shape, "Validation:", UXv.shape, "Test:", UXt.shape)

idx = np.random.RandomState(42).permutation(len(UXtr))
UXtr_shuffled, Uytr_shuffled = UXtr[idx], Uytr[idx]



f = np.linspace(0.1, 1.0, 10)
orig_train, orig_val = train_val_curve(
    LogisticRegression(solver='liblinear', class_weight='balanced', C=1.0),
    Xtr, ytr, Xv, yv, f
)
up_tr, up_val = train_val_curve(
    LogisticRegression(solver='liblinear', class_weight='balanced', C=1.0),
    UXtr_shuffled, Uytr_shuffled, UXv, Uyv, f
)

plt.figure(figsize=(8,5))
plt.plot(f * len(Xtr), orig_train, 'o-', label='Train (Original)')
plt.plot(f * len(Xtr), orig_val, 'o--', label='Val (Original)')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Original Data: Train vs Validation Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show(block=True)

plt.figure(figsize=(8,5))
plt.plot(f * len(UXtr), up_tr, 's-', label='Train (Upsampled)')
plt.plot(f * len(UXtr), up_val, 's--', label='Val (Upsampled)')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Upsampled Data: Train vs Validation Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show(block=True)

clf  = LogisticRegression(
    solver='liblinear', max_iter=1000,
    class_weight='balanced', random_state=42
)
clf2 = LogisticRegression(
    solver='liblinear', max_iter=1000,
    class_weight='balanced', random_state=42
)

from sklearn.model_selection import cross_val_score

n_splits_cv = 5
cv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)

scores_original = cross_val_score(
    clf, X, y,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
print(f"\nOriginal data {n_splits_cv}-fold CV accuracy: "
      f"{scores_original.mean():.3f} ± {scores_original.std():.3f}")
scores_upsampled = cross_val_score(
    clf2, X_res, y_res,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)
print(f"Upsampled data {n_splits_cv}-fold CV accuracy: "
      f"{scores_upsampled.mean():.3f} ± {scores_upsampled.std():.3f}")




if Xtr.shape[0] > 0 and UXtr.shape[0] > 0:
    clf.fit(Xtr, ytr)
    clf2.fit(UXtr, Uytr)


def evaluate(model, Xtest, ytest, title, overall_y_classes):
    if Xtest.shape[0] == 0:
        print(f"\n== {title} ==")
        return None
        
    preds = model.predict(Xtest)
    prob  = model.predict_proba(Xtest)
    acc   = accuracy_score(ytest, preds)
    rpt   = classification_report(ytest, preds, zero_division=0, labels=overall_y_classes, target_names=[f"Class {i}" for i in overall_y_classes])
    
    ybin = label_binarize(ytest, classes=overall_y_classes)
    auc_score_val = None
    
    if ybin.shape[1] == 1 and prob.shape[1] == 2: 
        auc_score_val = roc_auc_score(ytest, prob[:, 1], average='weighted')
    elif ybin.shape[1] > 1 and prob.shape[1] == ybin.shape[1]: 
        auc_score_val = roc_auc_score(ybin, prob, multi_class='ovr', average='weighted')
    else: 
    
        try: 
            if prob.shape[1] == 2 : 
                 auc_score_val = roc_auc_score(ytest, prob[:,1], average='weighted')
            else: 
                 auc_score_val = roc_auc_score(ytest, prob, multi_class='ovr', average='weighted')
        except ValueError as e_auc:
            print(f"AUC calculation error for '{title}': {e_auc}")
            
    cm = confusion_matrix(ytest, preds, labels=overall_y_classes)
    
    print(f"\n== {title} ==")
    print("Accuracy:", acc)
    if auc_score_val is not None:
        print("AUC(w):", auc_score_val)
    print(rpt)
    return cm

overall_unique_classes = np.unique(y)
cm1 = evaluate(clf, Xt, yt, "Original Test", overall_unique_classes)
cm2 = evaluate(clf2, Xt, yt, "Upsampled→Original Test", overall_unique_classes)

if cm1 is not None and cm2 is not None:
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=overall_unique_classes).plot(ax=axs[0], cmap=plt.cm.Blues)
    axs[0].set_title("Orig")
    ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=overall_unique_classes).plot(ax=axs[1], cmap=plt.cm.Blues)
    axs[1].set_title("Up")
    plt.tight_layout()
    plt.show()







Xlc = np.vstack((Xtr, Xv)); ylc = np.hstack((ytr, yv))

if Xlc.shape[0] > 0:
    minc_lc = 0
    unique_ylc, counts_ylc = np.unique(ylc, return_counts=True)
    if len(counts_ylc) > 0:
        minc_lc = np.min(counts_ylc)
    
    splits_lc = 5
    if minc_lc < splits_lc and minc_lc >=2 : 
        splits_lc = minc_lc
    elif minc_lc < 2: 
        splits_lc = 1 

    if splits_lc > 1:
        ts, tr_sc, cv_sc = learning_curve(
            LogisticRegression(solver='liblinear', max_iter=1000,
                               class_weight='balanced', random_state=42),
            Xlc, ylc,
            train_sizes=np.linspace(0.1,1.0,10),
            cv=StratifiedKFold(n_splits=splits_lc, shuffle=True, random_state=42),
            scoring='accuracy', n_jobs=-1
        )
        tr_m = tr_sc.mean(axis=1); cv_m = cv_sc.mean(axis=1)
        tr_std = tr_sc.std(axis=1); cv_std = cv_sc.std(axis=1)
        
        plt.figure()
        plt.fill_between(ts, tr_m-tr_std, tr_m+tr_std, alpha=0.1, color="r")
        plt.fill_between(ts, cv_m-cv_std, cv_m+cv_std, alpha=0.1, color="g")
        plt.plot(ts, tr_m, 'o-', color="r", label="Training score")
        plt.plot(ts, cv_m, 'o-', color="g", label="Cross-validation score")
        plt.title("Learning Curve")
        plt.xlabel("Training examples"); plt.ylabel("Accuracy Score")
        plt.legend(loc="best"); plt.grid(); plt.show()

if Xt.shape[0] > 0 and UXt.shape[0] > 0 : 
    orig_recalls = recall_score(yt, clf.predict(Xt), labels=overall_unique_classes, average=None, zero_division=0)
    up_recalls   = recall_score(Uyt, clf2.predict(UXt), labels=overall_unique_classes, average=None, zero_division=0)
    
    class_tick_labels = [str(c) for c in overall_unique_classes]
    x_indices = np.arange(len(class_tick_labels)); width = 0.35

    plt.figure(figsize=(max(8, len(class_tick_labels) * 1.5), 5)) 
    plt.bar(x_indices - width/2, orig_recalls, width, label='Original Test')
    plt.bar(x_indices + width/2, up_recalls,   width, label='Upsampled Test')
    plt.xticks(x_indices, class_tick_labels)
    plt.ylim(0,1)
    plt.xlabel('Class label'); plt.ylabel('Recall')
    plt.title('Per‑Class Recall: Before vs. After SMOTE')
    plt.legend(); plt.tight_layout(); plt.show()



def print_top_base_features(model_clf, f_names, ohe_base_names, n_top=5):

    coefs = model_clf.coef_

    model_classes = model_clf.classes_ 

    idx_to_base_feature_map = []
    for name in f_names:
        matched_base = next((base for base in ohe_base_names if name.startswith(base + '_')), None)
        idx_to_base_feature_map.append(matched_base if matched_base else name)

    if coefs.shape[0] == 1:
        agg_abs_weights = {}
        current_coefs = coefs[0]
        for feature_idx, base_feature_name in enumerate(idx_to_base_feature_map):
            agg_abs_weights.setdefault(base_feature_name, 0.0)
            agg_abs_weights[base_feature_name] += abs(current_coefs[feature_idx])
        
        top_features = sorted(agg_abs_weights.items(), key=lambda kv: kv[1], reverse=True)[:n_top]
        print(f"\nTop {n_top} dominant features:")
        for base_name, importance_val in top_features:
            print(f"  {base_name}: {importance_val:.3f}")
    else: 
        for i, class_label in enumerate(model_classes):
            agg_abs_weights = {}
            current_coefs = coefs[i]
            for feature_idx, base_feature_name in enumerate(idx_to_base_feature_map):
                agg_abs_weights.setdefault(base_feature_name, 0.0)
                agg_abs_weights[base_feature_name] += abs(current_coefs[feature_idx])
            
            top_features = sorted(agg_abs_weights.items(), key=lambda kv: kv[1], reverse=True)[:n_top]
            print(f"\nClass {class_label} top {n_top} dominant features: ")
            for base_name, importance_val in top_features:
                print(f"  {base_name}: {importance_val:.3f}")


print_top_base_features(clf, feature_names, ohe_input_feature_names) 
print_top_base_features(clf2, feature_names, ohe_input_feature_names)

if cm1 is not None and cm2 is not None:
    cm1_sum_axis1 = cm1.sum(axis=1, keepdims=True)
    cm1_pct = np.divide(cm1.astype(float), cm1_sum_axis1, out=np.zeros_like(cm1.astype(float)), where=cm1_sum_axis1!=0)
    
    cm2_sum_axis1 = cm2.sum(axis=1, keepdims=True)
    cm2_pct = np.divide(cm2.astype(float), cm2_sum_axis1, out=np.zeros_like(cm2.astype(float)), where=cm2_sum_axis1!=0)

    fig, axs = plt.subplots(1, 2, figsize=(14,6), sharey=True)
    sns.heatmap(cm1_pct, annot=True, fmt='.2f',
                xticklabels=overall_unique_classes, yticklabels=overall_unique_classes,
                cmap='Blues', ax=axs[0], vmin=0, vmax=1) 
    axs[0].set_title('Orig')
    axs[0].set_xlabel('Predicted label')
    axs[0].set_ylabel('True label')

    sns.heatmap(cm2_pct, annot=True, fmt='.2f',
                xticklabels=overall_unique_classes, yticklabels=overall_unique_classes,
                cmap='Blues', ax=axs[1], vmin=0, vmax=1) 
    axs[1].set_title('Up')
    axs[1].set_xlabel('Predicted label')

    plt.suptitle('Normalized Confusion Matrices (Rows Sum to 1)')
    plt.tight_layout(rect=[0,0,1,0.95]) 
    plt.show()

