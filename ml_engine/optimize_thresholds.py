
import pandas as pd

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from temporal_analysis import TemporalDriftAnalyzer


# ==========================================
# 1. LOAD DATASET AND MODEL
# ==========================================

df = pd.read_csv("dataset/trajectories.csv")

print("\n========== INTENTDRIFT OPTIMIZATION ==========")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 2. SPLIT BY TRAJECTORY
# ==========================================

train_ids = ["G001", "G002", "G003", "G004", "G005", "G006"]
test_ids = ["G007", "G008"]

train_df = df[df["Goal_ID"].isin(train_ids)]
test_df = df[df["Goal_ID"].isin(test_ids)]


# ==========================================
# 3. PRECOMPUTE SEMANTIC SIMILARITIES
# ==========================================

def prepare_data(data):

    prepared = []

    for goal_id, group in data.groupby("Goal_ID", sort=False):

        goal = group.iloc[0]["Original_Goal"]

        goal_embedding = model.encode(
            goal,
            convert_to_tensor=True
        )

        actions = []

        for _, row in group.iterrows():

            action_embedding = model.encode(
                row["Agent_Action"],
                convert_to_tensor=True
            )

            similarity = util.cos_sim(
                goal_embedding,
                action_embedding
            ).item()

            actions.append({
                "actual": row["Drift_Label"],
                "similarity": similarity,
                "action": row["Agent_Action"]
            })

        prepared.append(actions)

    return prepared


train_data = prepare_data(train_df)
test_data = prepare_data(test_df)


# ==========================================
# 4. TEMPORAL PREDICTION
# ==========================================

def predict_temporal(data, drift_threshold, score_threshold):

    actual = []
    predicted = []

    for trajectory in data:

        analyzer = TemporalDriftAnalyzer(
            window_size=3,
            drift_threshold=drift_threshold
        )

        for item in trajectory:

            drift = 1 - item["similarity"]

            result = analyzer.analyze(drift)

            prediction = (
                "Drift"
                if result["drift_score"] >= score_threshold
                else "Aligned"
            )

            actual.append(item["actual"])
            predicted.append(prediction)

    return actual, predicted


def calculate_metrics(actual, predicted):

    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted,
            pos_label="Drift",
            zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted,
            pos_label="Drift",
            zero_division=0
        ),
        "F1_Score": f1_score(
            actual, predicted,
            pos_label="Drift",
            zero_division=0
        )
    }


# ==========================================
# 5. SEARCH PARAMETER COMBINATIONS
# ==========================================

drift_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]

score_thresholds = [
    0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65
]

experiments = []

best_configuration = None
best_f1 = -1

print("\nSearching parameter combinations...")

for drift_threshold in drift_thresholds:

    for score_threshold in score_thresholds:

        actual, predicted = predict_temporal(
            train_data,
            drift_threshold,
            score_threshold
        )

        metrics = calculate_metrics(actual, predicted)

        experiment = {
            "Drift_Threshold": drift_threshold,
            "Score_Threshold": score_threshold,
            **metrics
        }

        experiments.append(experiment)

        if metrics["F1_Score"] > best_f1:

            best_f1 = metrics["F1_Score"]
            best_configuration = experiment


# ==========================================
# 6. DISPLAY BEST CONFIGURATION
# ==========================================

print("\n========== BEST CONFIGURATION ==========")

print(best_configuration)

# Save all experiments
pd.DataFrame(experiments).to_csv(
    "dataset/tuning_results.csv",
    index=False
)


# ==========================================
# 7. TEST ON UNSEEN TRAJECTORIES
# ==========================================

print("\n========== HELD-OUT TEST RESULTS ==========")

best_drift = best_configuration["Drift_Threshold"]
best_score = best_configuration["Score_Threshold"]

actual, predicted = predict_temporal(
    test_data,
    best_drift,
    best_score
)

test_metrics = calculate_metrics(actual, predicted)

print("\nOptimized IntentDrift:")

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")


# ==========================================
# 8. BASELINE TEST COMPARISON
# ==========================================

baseline_actual = []
baseline_predictions = []

for trajectory in test_data:

    for item in trajectory:

        baseline_actual.append(item["actual"])

        prediction = (
            "Drift"
            if item["similarity"] < 0.45
            else "Aligned"
        )

        baseline_predictions.append(prediction)


baseline_metrics = calculate_metrics(
    baseline_actual,
    baseline_predictions
)

print("\nBaseline Cosine Similarity:")

for metric, value in baseline_metrics.items():
    print(f"{metric}: {value:.4f}")


# ==========================================
# 9. SAVE FINAL TEST RESULTS
# ==========================================

comparison = pd.DataFrame([
    {"Model": "Baseline", **baseline_metrics},
    {"Model": "Optimized IntentDrift", **test_metrics}
])

comparison.to_csv(
    "dataset/optimized_comparison.csv",
    index=False
)

print("\n========== OPTIMIZATION COMPLETE ==========")

print("\nSaved files:")
print("dataset/tuning_results.csv")
print("dataset/optimized_comparison.csv")
