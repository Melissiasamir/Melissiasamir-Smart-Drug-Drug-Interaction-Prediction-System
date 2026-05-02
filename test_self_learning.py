#!/usr/bin/env python
"""Smoke tests for the self-learning module."""

import numpy as np
import pandas as pd

from utils.self_learning import SelfLearningConfig, SelfLearningEngine


np.random.seed(42)

config = SelfLearningConfig(sample_collection_threshold=5, persist_to_csv=False)
engine = SelfLearningEngine(config)
print(f"PASS: engine initialized with threshold={config.sample_collection_threshold}")

feature_columns = [f"feature_{i}" for i in range(10)]

for i in range(6):
    features = pd.DataFrame([np.random.randn(10)], columns=feature_columns)
    confidence = np.random.uniform(0.5, 0.95) if i % 2 == 0 else np.random.uniform(0.3, 0.6)
    engine.collect_prediction(
        drug_a=f"Drug_A_{i}",
        drug_b=f"Drug_B_{i}",
        features=features,
        prediction_label="Medium Risk",
        confidence=confidence,
        prediction_probabilities={"Low Risk": 0.3, "Medium Risk": 0.5, "High Risk": 0.2},
    )

status = engine.collector.get_collection_status()
assert status["total_collected"] == 6
assert status["uncertain_samples"] > 0
assert engine.should_trigger_learning()
print(f"PASS: collected {status['total_collected']} samples")
print(f"PASS: detected {status['uncertain_samples']} uncertain samples")

learning_df = engine.prepare_learning_dataset()
assert learning_df is not None
assert len(learning_df) > status["total_collected"]
assert set(feature_columns).issubset(learning_df.columns)
print(f"PASS: learning dataset prepared with {len(learning_df)} rows")

event = engine.trigger_learning_event()
assert event["triggered"]
assert event["learning_dataset_size"] == len(learning_df)
assert not engine.should_trigger_learning()
print("PASS: learning event triggered once and waits for a fresh batch")

summary = engine.get_summary()
assert summary["learning_events_triggered"] == 1
print("PASS: self-learning summary is consistent")

print("ALL SELF-LEARNING TESTS PASSED")
