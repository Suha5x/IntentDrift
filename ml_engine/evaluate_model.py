
import pandas as pd

from sentence_transformers import SentenceTransformer, util

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from temporal_analysis import TemporalDriftAnalyzer


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n========== INTENTDRIFT MODEL EVALUATION ==========")

df = pd.read_csv("dataset/trajectories.csv")

print("\nDataset loaded successfully!")
print("Total records:", len(df))


# ==========================================
# 2. LOAD SENTENCE TRANSFORMER
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 3. INITIALIZE PREDICTION STORAGE
# ==========================================

actual_labels = []
baseline_predictions = []
intentdrift_predictions = []

results = []

# Baseline threshold
similarity_threshold = 0.45


# ==========================================
# 4. PROCESS EACH TRAJECTORY
# ==========================================

for goal_id, group in df.groupby("Goal_ID", sort=False):

    print(f"\nProcessing trajectory: {goal_id}")

    goal = group.iloc[0]["Original_Goal"]

    # New analyzer for each independent trajectory
    analyzer = TemporalDriftAnalyzer(
        window_size=3,
        drift_threshold=0.45
    )

    goal_embedding = model.encode(
        goal,
        convert_to_tensor=True
    )

    for _, row in group.iterrows():

        action = row["Agent_Action"]

        actual_label = row["Drift_Label"]

        # Generate action embedding
        action_embedding = model.encode(
            action,
            convert_to_tensor=True
        )

        # Calculate semantic similarity
        similarity = util.cos_sim(
            goal_embedding,
            action_embedding
        ).item()

        # Instantaneous drift
        instantaneous_drift = 1 - similarity

        # ----------------------------------
        # BASELINE: Similarity only
        # ----------------------------------

        if similarity < similarity_threshold:
            baseline_prediction = "Drift"
        else:
            baseline_prediction = "Aligned"

        # ----------------------------------
        # INTENTDRIFT: Temporal detection
        # ----------------------------------

        analysis = analyzer.analyze(
            instantaneous_drift
        )

        if analysis["status"] in ["WARNING", "HIGH DRIFT"]:
            intentdrift_prediction = "Drift"
        else:
            intentdrift_prediction = "Aligned"

        # Store predictions
        actual_labels.append(actual_label)

        baseline_predictions.append(
            baseline_prediction
        )

        intentdrift_predictions.append(
            intentdrift_prediction
        )

        results.append({
            "Goal_ID": goal_id,
            "Action_ID": row["Action_ID"],
            "Agent_Action": action,
            "Actual_Label": actual_label,
            "Similarity": round(similarity, 4),
            "Instantaneous_Drift": round(
                instantaneous_drift, 4
            ),
            "Drift_Score": round(
                analysis["drift_score"], 4
            ),
            "IntentDrift_Status": analysis["status"],
            "Baseline_Prediction": baseline_prediction,
            "IntentDrift_Prediction": intentdrift_prediction
        })


# ==========================================
# 5. EVALUATION FUNCTION
# ==========================================

def evaluate(name, actual, predicted):

    print(f"\n========== {name} ==========")

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual, predicted,
        pos_label="Drift",
        zero_division=0
    )

    recall = recall_score(
        actual, predicted,
        pos_label="Drift",
        zero_division=0
    )

    f1 = f1_score(
        actual, predicted,
        pos_label="Drift",
        zero_division=0
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=["Aligned", "Drift"]
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=["Aligned", "Drift"],
            zero_division=0
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }


# ==========================================
# 6. COMPARE BOTH APPROACHES
# ==========================================

baseline_metrics = evaluate(
    "BASELINE - COSINE SIMILARITY",
    actual_labels,
    baseline_predictions
)

intentdrift_metrics = evaluate(
    "INTENTDRIFT - TEMPORAL DETECTION",
    actual_labels,
    intentdrift_predictions
)


# ==========================================
# 7. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "dataset/evaluation_results.csv",
    index=False
)

comparison_df = pd.DataFrame([
    baseline_metrics,
    intentdrift_metrics
])

comparison_df.to_csv(
    "dataset/model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")

print(
    comparison_df.to_string(index=False)
)

print("\nEvaluation results saved successfully!")

print("1. dataset/evaluation_results.csv")
print("2. dataset/model_comparison.csv")

print("\n========== EVALUATION COMPLETE ==========")
