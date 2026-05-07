"""Self-learning support for Smart Drug Risk Agent.

The module collects inference samples, detects uncertain predictions, augments
those uncertain samples, and can run a safe DBSCAN + FCM reclustering preview.
The preview is intentionally non-destructive: it never replaces the trained SVM
or the saved production pipeline unless a future explicit promotion step is
added.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class SelfLearningConfig:
    """Configuration for the self-learning module."""

    sample_collection_threshold: int = 20
    uncertainty_confidence_threshold: float = 0.6
    augmentation_noise_std: float = 0.05
    persist_to_csv: bool = True
    csv_path: Path | str = "data/self_learning_samples.csv"
    log_actions: bool = True


class SelfLearningCollector:
    """Collects and manages samples for self-learning."""

    def __init__(self, config: SelfLearningConfig | None = None):
        self.config = config or SelfLearningConfig()
        self.collected_samples: list[dict] = []
        self.uncertain_samples: list[dict] = []
        self.learning_triggered_count = 0
        self.session_start_time = datetime.now()
        self._persisted_sample_count = 0

    def collect_sample(
        self,
        drug_a: str,
        drug_b: str,
        features: np.ndarray | pd.Series | pd.DataFrame,
        prediction_label: str,
        confidence: float,
        prediction_probabilities: dict[str, float],
        svm_gap: float | None = None,
        fuzzy_membership: float | None = None,
        agentic_score: float | None = None,
        action_route: str | None = None,
    ) -> dict[str, object]:
        """Collect a feature row and prediction metadata from one inference."""
        feature_names = None
        if isinstance(features, pd.DataFrame):
            feature_names = [str(col) for col in features.columns]
            feature_values = features.values.flatten()
        elif isinstance(features, pd.Series):
            feature_names = [str(idx) for idx in features.index]
            feature_values = features.values
        else:
            feature_values = np.array(features).flatten()

        sample = {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "prediction_label": prediction_label,
            "confidence": float(confidence),
            "probabilities": prediction_probabilities,
            "svm_gap": svm_gap,
            "fuzzy_membership": fuzzy_membership,
            "agentic_score": agentic_score,
            "action_route": action_route,
            "features": feature_values.astype(float),
            "feature_names": feature_names,
            "feature_count": len(feature_values),
            "timestamp": datetime.now().isoformat(),
            "is_uncertain": confidence < self.config.uncertainty_confidence_threshold,
        }

        self.collected_samples.append(sample)

        if sample["is_uncertain"]:
            self.uncertain_samples.append(sample)
            if self.config.log_actions:
                logger.info(
                    "Uncertain sample collected: %s-%s (%s, confidence=%.2f%%)",
                    drug_a,
                    drug_b,
                    prediction_label,
                    confidence * 100,
                )
        elif self.config.log_actions:
            logger.info(
                "Sample collected: %s-%s (%s, confidence=%.2f%%)",
                drug_a,
                drug_b,
                prediction_label,
                confidence * 100,
            )

        return sample

    def get_collection_status(self) -> dict[str, object]:
        """Return current collection counts and trigger state."""
        return {
            "total_collected": len(self.collected_samples),
            "uncertain_samples": len(self.uncertain_samples),
            "collection_threshold": self.config.sample_collection_threshold,
            "learning_triggered_count": self.learning_triggered_count,
            "can_trigger_learning": self._can_trigger_learning(),
            "session_start": self.session_start_time.isoformat(),
        }

    def _can_trigger_learning(self) -> bool:
        """Require a fresh threshold-sized batch for each learning event."""
        next_threshold = self.config.sample_collection_threshold * (self.learning_triggered_count + 1)
        return len(self.collected_samples) >= next_threshold


class DataAugmentor:
    """Augments collected data with small synthetic numeric variations."""

    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std

    def augment_sample(self, sample: dict, num_augmentations: int = 2) -> list[dict]:
        """Create augmented versions of a sample, including the original."""
        augmented_list = [sample.copy()]
        features = np.array(sample["features"], dtype=float)

        for i in range(num_augmentations):
            augmented_sample = sample.copy()
            augmented_sample["features"] = np.maximum(
                features + np.random.normal(0, self.noise_std, features.shape),
                0,
            )
            augmented_sample["augmentation_round"] = i + 1
            augmented_sample["is_augmented"] = True
            augmented_list.append(augmented_sample)

        return augmented_list

    def augment_batch(
        self, samples: list[dict], num_augmentations_per_sample: int = 2
    ) -> list[dict]:
        """Augment a batch of samples."""
        augmented_batch = []
        for sample in samples:
            augmented_batch.extend(self.augment_sample(sample, num_augmentations_per_sample))
        return augmented_batch


class SelfLearningEngine:
    """Coordinates data collection, augmentation, persistence, and safe triggers."""

    def __init__(self, config: SelfLearningConfig | None = None):
        self.config = config or SelfLearningConfig()
        self.collector = SelfLearningCollector(self.config)
        self.augmentor = DataAugmentor(self.config.augmentation_noise_std)
        self.learning_history: list[dict] = []

    def collect_prediction(
        self,
        drug_a: str,
        drug_b: str,
        features: np.ndarray | pd.Series | pd.DataFrame,
        prediction_label: str,
        confidence: float,
        prediction_probabilities: dict[str, float],
        svm_gap: float | None = None,
        fuzzy_membership: float | None = None,
        agentic_score: float | None = None,
        action_route: str | None = None,
    ) -> None:
        """Record a prediction for future self-learning."""
        self.collector.collect_sample(
            drug_a=drug_a,
            drug_b=drug_b,
            features=features,
            prediction_label=prediction_label,
            confidence=confidence,
            prediction_probabilities=prediction_probabilities,
            svm_gap=svm_gap,
            fuzzy_membership=fuzzy_membership,
            agentic_score=agentic_score,
            action_route=action_route,
        )

    def should_trigger_learning(self) -> bool:
        """Return True when a new threshold-sized batch is available."""
        return self.collector._can_trigger_learning()

    def prepare_learning_dataset(self) -> pd.DataFrame | None:
        """Return original samples plus augmented uncertain samples."""
        if not self.should_trigger_learning():
            return None

        all_samples = self.collector.collected_samples.copy()

        if self.collector.uncertain_samples:
            uncertain_augmented = self.augmentor.augment_batch(
                self.collector.uncertain_samples, num_augmentations_per_sample=3
            )
            all_samples.extend(uncertain_augmented)
            if self.config.log_actions:
                logger.info(
                    "Augmented %s uncertain samples into %s uncertain/augmented rows",
                    len(self.collector.uncertain_samples),
                    len(uncertain_augmented),
                )

        learning_data = []
        for sample in all_samples:
            record = {
                "drug_a": sample["drug_a"],
                "drug_b": sample["drug_b"],
                "prediction_label": sample["prediction_label"],
                "confidence": sample["confidence"],
                "is_uncertain": sample["is_uncertain"],
                "is_augmented": sample.get("is_augmented", False),
                "timestamp": sample["timestamp"],
            }
            feature_names = sample.get("feature_names") or [
                f"feature_{i}" for i in range(len(sample["features"]))
            ]
            for feature_name, feat_val in zip(feature_names, sample["features"]):
                record[str(feature_name)] = float(feat_val)

            learning_data.append(record)

        df = pd.DataFrame(learning_data)

        if self.config.log_actions:
            logger.info("Prepared learning dataset: %s samples total", len(df))

        return df

    def trigger_learning_event(
        self,
        scaler=None,
        feature_columns: list[str] | None = None,
        run_reclustering_preview: bool = False,
        baseline_clustering_metadata: dict | None = None,
        baseline_svm_gap: float | None = None,
    ) -> dict[str, object]:
        """Trigger learning metadata and optionally a non-destructive reclustering preview."""
        status = self.collector.get_collection_status()

        if not self.should_trigger_learning():
            return {
                "triggered": False,
                "reason": "Not enough new samples collected",
                "status": status,
            }

        learning_df = self.prepare_learning_dataset()
        reclustering_metadata = None
        pseudo_label_counts = None
        drift_decision = None

        if run_reclustering_preview and learning_df is not None and scaler is not None and feature_columns:
            from utils.clustering import generate_pseudo_labels

            missing_columns = [col for col in feature_columns if col not in learning_df.columns]
            if missing_columns:
                logger.warning(
                    "Skipped reclustering preview because learning data is missing feature columns: %s",
                    missing_columns,
                )
            else:
                original_features = learning_df[feature_columns].astype(float).dropna()
                if len(original_features) >= 3:
                    scaled_features = pd.DataFrame(
                        scaler.transform(original_features),
                        columns=feature_columns,
                        index=original_features.index,
                    )
                    pseudo_labels, reclustering_metadata = generate_pseudo_labels(
                        scaled_features,
                        original_features,
                    )
                    pseudo_label_counts = pseudo_labels["pseudo_label"].value_counts().to_dict()
                    if baseline_clustering_metadata is not None:
                        from utils.drift_detection import evaluate_self_learning_drift

                        recent_gaps = [
                            float(sample["svm_gap"])
                            for sample in self.collector.collected_samples
                            if sample.get("svm_gap") is not None
                        ]
                        drift_decision = evaluate_self_learning_drift(
                            baseline_gap=baseline_svm_gap,
                            recent_gaps=recent_gaps,
                            old_clustering_metadata=baseline_clustering_metadata,
                            new_clustering_metadata=reclustering_metadata,
                        )
                else:
                    logger.warning("Skipped reclustering preview because fewer than 3 complete rows are available")

        event = {
            "triggered": True,
            "timestamp": datetime.now().isoformat(),
            "samples_collected": len(self.collector.collected_samples),
            "uncertain_samples": len(self.collector.uncertain_samples),
            "learning_dataset_size": len(learning_df) if learning_df is not None else 0,
            "reclustering_preview_ran": reclustering_metadata is not None,
            "reclustering_metadata": reclustering_metadata,
            "pseudo_label_counts": pseudo_label_counts,
            "drift_decision": drift_decision,
            "status": status,
            "learning_df": learning_df,
        }

        self.collector.learning_triggered_count += 1
        self.learning_history.append(
            {key: value for key, value in event.items() if key != "learning_df"}
        )

        if self.config.log_actions:
            logger.info(
                "Learning triggered. Event #%s: %s original, %s uncertain, %s total, reclustering_preview=%s",
                self.collector.learning_triggered_count,
                event["samples_collected"],
                event["uncertain_samples"],
                event["learning_dataset_size"],
                event["reclustering_preview_ran"],
            )

        return event

    def persist_samples(self, filepath: Path | str | None = None) -> bool:
        """Append only new collected samples to CSV."""
        if not self.config.persist_to_csv:
            return False

        filepath = Path(filepath or self.config.csv_path)
        new_samples = self.collector.collected_samples[self.collector._persisted_sample_count :]
        if not new_samples:
            return False

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            save_data = []
            for sample in new_samples:
                record = {
                    "drug_a": sample["drug_a"],
                    "drug_b": sample["drug_b"],
                    "prediction_label": sample["prediction_label"],
                    "confidence": sample["confidence"],
                    "is_uncertain": sample["is_uncertain"],
                    "svm_gap": sample.get("svm_gap"),
                    "fuzzy_membership": sample.get("fuzzy_membership"),
                    "agentic_score": sample.get("agentic_score"),
                    "action_route": sample.get("action_route"),
                    "timestamp": sample["timestamp"],
                }
                feature_names = sample.get("feature_names") or [
                    f"feature_{i}" for i in range(len(sample["features"]))
                ]
                for feature_name, feat_val in zip(feature_names, sample["features"]):
                    record[str(feature_name)] = float(feat_val)

                save_data.append(record)

            new_df = pd.DataFrame(save_data)
            if filepath.exists():
                existing_columns = list(pd.read_csv(filepath, nrows=0).columns)
                if existing_columns != list(new_df.columns):
                    existing_df = pd.read_csv(filepath)
                    combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
                    combined.to_csv(filepath, index=False)
                else:
                    new_df.to_csv(filepath, mode="a", header=False, index=False)
            else:
                new_df.to_csv(filepath, index=False)
            self.collector._persisted_sample_count = len(self.collector.collected_samples)

            if self.config.log_actions:
                logger.info("Persisted %s new samples to %s", len(new_samples), filepath)

            return True
        except Exception as exc:
            logger.error("Failed to persist samples: %s", exc)
            return False

    def load_samples(self, filepath: Path | str | None = None) -> bool:
        """Load previously persisted samples from CSV."""
        filepath = Path(filepath or self.config.csv_path)

        if not filepath.exists():
            return False

        try:
            df = pd.read_csv(filepath)
            metadata_cols = {
                "drug_a",
                "drug_b",
                "prediction_label",
                "confidence",
                "is_uncertain",
                "timestamp",
                "svm_gap",
                "fuzzy_membership",
                "agentic_score",
                "action_route",
            }
            feature_cols = [col for col in df.columns if col not in metadata_cols]

            for _, row in df.iterrows():
                features = row[feature_cols].astype(float).values
                sample = {
                    "drug_a": row["drug_a"],
                    "drug_b": row["drug_b"],
                    "prediction_label": row["prediction_label"],
                    "confidence": float(row["confidence"]),
                    "probabilities": {},
                    "svm_gap": None if pd.isna(row.get("svm_gap")) else float(row.get("svm_gap")),
                    "fuzzy_membership": None if pd.isna(row.get("fuzzy_membership")) else float(row.get("fuzzy_membership")),
                    "agentic_score": None if pd.isna(row.get("agentic_score")) else float(row.get("agentic_score")),
                    "action_route": None if pd.isna(row.get("action_route")) else str(row.get("action_route")),
                    "features": features,
                    "feature_names": feature_cols,
                    "feature_count": len(features),
                    "timestamp": row["timestamp"],
                    "is_uncertain": str(row["is_uncertain"]).lower() == "true",
                }

                self.collector.collected_samples.append(sample)
                if sample["is_uncertain"]:
                    self.collector.uncertain_samples.append(sample)

            self.collector._persisted_sample_count = len(self.collector.collected_samples)

            if self.config.log_actions:
                logger.info("Loaded %s samples from %s", len(self.collector.collected_samples), filepath)

            return True
        except Exception as exc:
            logger.error("Failed to load samples: %s", exc)
            return False

    def get_summary(self) -> dict[str, object]:
        """Return a compact state summary for the dashboard."""
        return {
            "total_samples_collected": len(self.collector.collected_samples),
            "uncertain_samples_count": len(self.collector.uncertain_samples),
            "learning_events_triggered": self.collector.learning_triggered_count,
            "can_trigger_now": self.should_trigger_learning(),
            "collection_threshold": self.config.sample_collection_threshold,
            "confidence_threshold": self.config.uncertainty_confidence_threshold,
            "session_active_since": self.collector.session_start_time.isoformat(),
            "learning_history_count": len(self.learning_history),
            "recent_mean_svm_gap": self._recent_mean("svm_gap"),
            "recent_mean_agentic_score": self._recent_mean("agentic_score"),
            "latest_action_route": self.collector.collected_samples[-1].get("action_route")
            if self.collector.collected_samples
            else None,
        }

    def _recent_mean(self, key: str, window: int = 20) -> float | None:
        values = [
            float(sample[key])
            for sample in self.collector.collected_samples[-window:]
            if sample.get(key) is not None
        ]
        return float(np.mean(values)) if values else None
