
from collections import deque
import numpy as np


class TemporalDriftAnalyzer:

    def __init__(
        self,
        window_size=3,
        drift_threshold=0.45,
        alpha=0.4,
        beta=0.3,
        gamma=0.3
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold

        # Weights for Drift Score
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Store recent drift values
        self.drift_history = deque(maxlen=window_size)

        # Store all drift values
        self.all_drifts = []

    def analyze(self, instantaneous_drift):

        self.drift_history.append(instantaneous_drift)
        self.all_drifts.append(instantaneous_drift)

        # 1. Recent average drift
        recent_mean = np.mean(self.drift_history)

        # 2. Persistence of high drift
        high_drift_count = sum(
            value > self.drift_threshold
            for value in self.drift_history
        )

        persistence = high_drift_count / len(self.drift_history)

        # 3. Cumulative deviation
        cumulative = np.mean(self.all_drifts)

        # 4. Combined Drift Score
        drift_score = (
            self.alpha * recent_mean
            + self.beta * persistence
            + self.gamma * cumulative
        )

        # 5. Classification
        if drift_score < 0.25:
            status = "ALIGNED"

        elif drift_score < 0.40:
            status = "MONITORING"

        elif drift_score < 0.55:
            status = "WARNING"

        else:
            status = "HIGH DRIFT"

        # 6. Intervention recommendation
        if status == "HIGH DRIFT":
            intervention = "PAUSE AND REQUEST USER CONFIRMATION"

        elif status == "WARNING":
            intervention = "WARN USER"

        elif status == "MONITORING":
            intervention = "CONTINUE MONITORING"

        else:
            intervention = "CONTINUE AGENT"

        # 7. Return results
        return {
            "recent_mean": float(recent_mean),
            "persistence": float(persistence),
            "cumulative": float(cumulative),
            "drift_score": float(drift_score),
            "status": status,
            "intervention": intervention
        }