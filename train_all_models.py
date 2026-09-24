import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
warnings.filterwarnings('ignore')
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
os.makedirs('models', exist_ok=True)
data_file = 'preprocessed_loan_default.csv' if os.path.exists('preprocessed_loan_default.csv') else 'Loan_default.csv'
df = pd.read_csv(data_file)
if 'Default' in df.columns and 'LoanDefault' not in df.columns:
    df['LoanDefault'] = df['Default']
binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0, 1: 1, 0: 0}).fillna(0).astype(int)
FEATURE_NAMES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
    'HasMortgage', 'HasDependents'
]
X = df[FEATURE_NAMES]
y = df['LoanDefault'].astype(int)
class CustomLogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.05, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        y_arr = np.array(y, dtype=np.float64)
        X_arr = np.array(X, dtype=np.float64)
        for i in range(self.n_iterations):
            linear_model = np.dot(X_arr, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_predicted, epsilon, 1 - epsilon)
            loss = -np.mean(y_arr * np.log(y_pred_clipped) + (1 - y_arr) * np.log(1 - y_pred_clipped))
            self.loss_history.append(loss)
            dw = (1 / n_samples) * np.dot(X_arr.T, (y_predicted - y_arr))
            db = (1 / n_samples) * np.sum(y_predicted - y_arr)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    def predict_proba(self, X):
        linear_model = np.dot(np.array(X), self.weights) + self.bias
        return self._sigmoid(linear_model)
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
df.to_csv('preprocessed_loan_default.csv', index=False)
joblib.dump(scaler, 'models/scaler.pkl')
model_metrics = {}
train_test_comparison = {}
def evaluate_and_record(name, model, X_tr, y_tr, X_te, y_te, train_time=0.0):
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_te)
        if y_test_prob.ndim == 2:
            y_test_prob = y_test_prob[:, 1]
    elif hasattr(model, "decision_function"):
        y_test_prob = model.decision_function(X_te)
    else:
        y_test_prob = y_test_pred
    train_acc = accuracy_score(y_tr, y_train_pred)
    test_acc = accuracy_score(y_te, y_test_pred)
    prec = precision_score(y_te, y_test_pred, zero_division=0)
    rec = recall_score(y_te, y_test_pred)
    f1 = f1_score(y_te, y_test_pred)
    roc_auc = roc_auc_score(y_te, y_test_prob)
    model_metrics[name] = {
        "Accuracy": test_acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC_AUC": roc_auc,
        "Time (s)": round(train_time, 2)
    }
    train_test_comparison[name] = {
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Overfit Gap": train_acc - test_acc
    }
    print(f"\n=== {name} ===")
    print(f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Gap: {(train_acc - test_acc)*100:+.2f}%")
    print(f"Recall: {rec*100:.2f}% | Precision: {prec*100:.2f}% | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | Time: {train_time:.2f}s")
    return y_test_pred, y_test_prob
scratch_lr = CustomLogisticRegressionFromScratch(learning_rate=0.1, n_iterations=600)
t0 = time.time()
scratch_lr.fit(X_train_scaled, y_train)
scratch_time = time.time() - t0
scratch_pred = scratch_lr.predict(X_test_scaled)
scratch_prob = scratch_lr.predict_proba(X_test_scaled)
print(f"Scratch Accuracy: {accuracy_score(y_test, scratch_pred)*100:.2f}% | ROC-AUC: {roc_auc_score(y_test, scratch_prob):.4f} | Time: {scratch_time:.2f}s")
t0 = time.time()
sk_lr = LogisticRegression(max_iter=1000, random_state=42)
sk_lr.fit(X_train_scaled, y_train)
sk_time = time.time() - t0
sk_pred, sk_prob = evaluate_and_record("Logistic Regression", sk_lr, X_train_scaled, y_train, X_test_scaled, y_test, train_time=sk_time)
joblib.dump(sk_lr, 'models/logistic_regression.pkl')
t0 = time.time()
dt_model = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)
dt_time = time.time() - t0
dt_pred, dt_prob = evaluate_and_record("Decision Tree", dt_model, X_train, y_train, X_test, y_test, train_time=dt_time)
joblib.dump(dt_model, 'models/decision_tree.pkl')
sample_size = min(30000, len(X_train))
knn_idx = np.random.RandomState(42).choice(len(X_train_scaled), sample_size, replace=False)
X_tr_knn = X_train_scaled[knn_idx]
y_tr_knn = y_train.iloc[knn_idx]
t0 = time.time()
knn_model = KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=-1)
knn_model.fit(X_tr_knn, y_tr_knn)
knn_time = time.time() - t0
test_knn_idx = np.random.RandomState(42).choice(len(X_test_scaled), min(10000, len(X_test_scaled)), replace=False)
knn_pred, knn_prob = evaluate_and_record("K-Nearest Neighbors", knn_model, X_tr_knn, y_tr_knn, X_test_scaled[test_knn_idx], y_test.iloc[test_knn_idx], train_time=knn_time)
joblib.dump(knn_model, 'models/k_neighbours.pkl')
t0 = time.time()
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_time = time.time() - t0
rf_pred, rf_prob = evaluate_and_record("Random Forest", rf_model, X_train, y_train, X_test, y_test, train_time=rf_time)
joblib.dump(rf_model, 'models/random_forest.pkl', compress=3)
t0 = time.time()
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)
gnb_time = time.time() - t0
gnb_pred, gnb_prob = evaluate_and_record("Gaussian Naive Bayes", gnb_model, X_train_scaled, y_train, X_test_scaled, y_test, train_time=gnb_time)
joblib.dump(gnb_model, 'models/gaussian_nb.pkl')
joblib.dump(gnb_model, 'models/probabilistic.pkl')
t0 = time.time()
base_svc = LinearSVC(class_weight='balanced', random_state=42, max_iter=3000, dual=False)
svc_model = CalibratedClassifierCV(estimator=base_svc, cv=3)
svc_model.fit(X_train_scaled, y_train)
svc_time = time.time() - t0
svc_pred, svc_prob = evaluate_and_record("Support Vector Classification (SVC)", svc_model, X_train_scaled, y_train, X_test_scaled, y_test, train_time=svc_time)
joblib.dump(svc_model, 'models/svc.pkl')
joblib.dump(svc_model, 'models/svm.pkl')
t0 = time.time()
base_dt = DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)
bagging_model = BaggingClassifier(estimator=base_dt, n_estimators=60, random_state=42, n_jobs=-1)
bagging_model.fit(X_train, y_train)
bag_time = time.time() - t0
bag_pred, bag_prob = evaluate_and_record("Bagging Classifier", bagging_model, X_train, y_train, X_test, y_test, train_time=bag_time)
joblib.dump(bagging_model, 'models/bagging.pkl')
t0 = time.time()
boosting_model = HistGradientBoostingClassifier(max_iter=160, class_weight='balanced', random_state=42, min_samples_leaf=20)
boosting_model.fit(X_train, y_train)
boost_time = time.time() - t0
boost_pred, boost_prob = evaluate_and_record("Gradient Boosting", boosting_model, X_train, y_train, X_test, y_test, train_time=boost_time)
joblib.dump(boosting_model, 'models/boosting.pkl')
joblib.dump(boosting_model, 'models/gradient_boosting.pkl')
t0 = time.time()
adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=42)
adaboost_model.fit(X_train, y_train)
ada_time = time.time() - t0
ada_pred, ada_prob = evaluate_and_record("AdaBoost", adaboost_model, X_train, y_train, X_test, y_test, train_time=ada_time)
joblib.dump(adaboost_model, 'models/adaboost.pkl')
joblib.dump(model_metrics, 'models/new_models_metrics.pkl')
results_df = pd.DataFrame(model_metrics).T
results_df = results_df.sort_values(by="ROC_AUC", ascending=False)
print("\n" + "=" * 80)
print("ALL MODEL BENCHMARK RESULTS")
print("=" * 80)
print(results_df.to_string())
print("\n" + "=" * 80)
print("SAVED ARTIFACTS IN models/")
print("=" * 80)
for fname in sorted(os.listdir('models')):
    fpath = os.path.join('models', fname)
    size_kb = os.path.getsize(fpath) / 1024
    if size_kb > 1024:
        print(f"  • models/{fname:<30} ({size_kb/1024:>6.2f} MB)")
    else:
        print(f"  • models/{fname:<30} ({size_kb:>6.1f} KB)")
