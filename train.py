import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc, r2_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
from itertools import cycle

#  Load Data 
LABEL_FILE_PATH = r"D:\Chetan\Msc 1\Final 2.0\label_list.txt"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.isfile(LABEL_FILE_PATH):
    raise FileNotFoundError(" Error: label_list.txt file not found.")

print(" Info: label_list.txt found.")

# Initialize feature & label lists
features = {
    "baseline_angle": [], "letter_size": [], "line_spacing": [],
    "word_spacing": [], "pen_pressure": [], "slant_angle": []
}
labels = []

# Read and process the label file
with open(LABEL_FILE_PATH, "r") as labels_file:
    for line in labels_file:
        content = line.split()
        for i, key in enumerate(features.keys()):
            features[key].append(float(content[i]) + np.random.uniform(-0.5, 0.5))  # Adding small noise
        labels.append(int(content[7]))  # Assuming t1 (Emotional Stability) as the target label

#  Prepare Feature Set 
X = np.array([
    [features["baseline_angle"][i], features["letter_size"][i], features["line_spacing"][i], 
     features["word_spacing"][i], features["pen_pressure"][i], features["slant_angle"][i]]
    for i in range(len(labels))
])

y = np.array(labels)

#  Data Scaling 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Train Unified Model 
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=42)
for train_idx, test_idx in sss.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# Train SVM Model
clf = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

#  Model Metrics 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(clf, X_scaled, y, cv=3, scoring='accuracy')

# Handle AUC Calculation for Multi-class
try:
    overall_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
except ValueError:
    overall_auc = "N/A (Only one class present in y_true)"

overall_conf_matrix = confusion_matrix(y_test, y_pred)

# Print Metrics
print("\n📊 **Overall Performance Metrics:**")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔄 Recall: {recall:.4f}")
print(f"📊 F1 Score: {f1:.4f}")
print(f"📈 R² Score: {r2:.4f}")
print(f"📈 CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"📈 AUC Score: {overall_auc}")
print(f"📌 Confusion Matrix:\n{overall_conf_matrix}")

# Per-Class Recall
unique_classes = np.unique(y)
print("\n🔍 Per-Class Recall:")
for cls in unique_classes:
    cls_recall = recall_score(y_test == cls, y_pred == cls, zero_division=1)
    print(f"  Class {cls}: {cls_recall:.4f}")

# ======================== Save Final Model & Scaler ========================
model_path = os.path.join(MODEL_DIR, "final_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

joblib.dump(clf, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n Final Model saved as: {model_path}")
print(f"Scaler saved as: {scaler_path}")

# ======================== Plot Confusion Matrix ========================
plt.figure(figsize=(6, 5))
class_labels = np.unique(y)
sns.heatmap(overall_conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Overall Confusion Matrix")
plt.show()

# ======================== Plot AUC ROC Curve ========================
if len(unique_classes) > 2:
    # One-vs-Rest ROC curves
    y_test_bin = label_binarize(y_test, classes=unique_classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])

    plt.figure(figsize=(8, 6))
    for i, color in zip(range(len(unique_classes)), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {unique_classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
else:
    print(" ROC Curve skipped: Not enough classes for multi-class AUC plot.")

print("\n Training Complete. Final model and scaler saved.")
