
from sentence_transformers import SentenceTransformer, util
from temporal_analysis import TemporalDriftAnalyzer

# Load embedding model
print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize temporal analyzer
analyzer = TemporalDriftAnalyzer(
    window_size=3,
    drift_threshold=0.45
)

# Original user goal
goal = "Find the cheapest laptop under 60000 rupees with 16GB RAM"

# Simulated agent actions
actions = [
    "Search laptops under 60000 rupees",
    "Filter laptops with 16GB RAM",
    "Compare laptop prices",
    "Check laptop processor specifications",
    "Search gaming laptops above 100000 rupees",
    "Browse gaming accessories",
    "Add an unrelated keyboard to cart"
]

# Generate goal embedding
goal_embedding = model.encode(
    goal,
    convert_to_tensor=True
)

print("\n========== INTENTDRIFT MONITOR ==========")
print("Original Goal:", goal)

for i, action in enumerate(actions, start=1):

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

    # Calculate instantaneous drift
    instantaneous_drift = 1 - similarity

    # Perform temporal analysis
    result = analyzer.analyze(instantaneous_drift)

    print("\n----------------------------------------")
    print(f"ACTION {i}: {action}")

    print(f"Semantic Similarity : {similarity:.4f}")
    print(f"Instantaneous Drift : {instantaneous_drift:.4f}")
    print(f"Recent Mean Drift   : {result['recent_mean']:.4f}")
    print(f"Persistence         : {result['persistence']:.2f}")
    print(f"Cumulative Drift    : {result['cumulative']:.4f}")

    print(f"FINAL DRIFT SCORE   : {result['drift_score']:.4f}")
    print(f"STATUS              : {result['status']}")
    print(f"INTERVENTION        : {result['intervention']}")

print("\n========== MONITORING COMPLETE ==========")
