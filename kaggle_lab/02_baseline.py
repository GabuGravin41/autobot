"""
Step 2 — train a real baseline and write submission.csv.

Not a dummy/constant-prediction baseline — actual feature engineering plus
a random forest, which typically lands ~0.77-0.78 public accuracy on
Titanic (top-tier hand-tuned solutions get to ~0.80-0.82; anything above
~0.85 on the public leaderboard is near-certainly leakage, since the test
labels for this competition are publicly known and some leaderboard
entries hardcode them - worth knowing before treating a suspiciously high
score as a benchmark to beat).

Requires: pandas, scikit-learn. If either is missing:
  pip install pandas scikit-learn
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Pclass"] = df["Pclass"]
    out["Sex"] = (df["Sex"] == "male").astype(int)
    out["Age"] = df["Age"].fillna(df["Age"].median())
    out["SibSp"] = df["SibSp"]
    out["Parch"] = df["Parch"]
    out["Fare"] = df["Fare"].fillna(df["Fare"].median())
    embarked = df["Embarked"].fillna("S")
    out["Embarked_C"] = (embarked == "C").astype(int)
    out["Embarked_Q"] = (embarked == "Q").astype(int)
    out["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    return out


X = featurize(train)
y = train["Survived"]
X_test = featurize(test)

model = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=2,
    random_state=42, n_jobs=-1,
)

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

model.fit(X, y)
preds = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": preds,
})
submission.to_csv("data/submission.csv", index=False)
print("Wrote data/submission.csv —", len(submission), "rows")
print(submission.head())
