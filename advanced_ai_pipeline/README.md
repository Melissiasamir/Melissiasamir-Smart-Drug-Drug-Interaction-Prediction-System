# Advanced AI Pipeline

This optional extension adds a plug-in layer to the Smart Drug-Drug Interaction Prediction System.
It is designed to be additive and non-invasive: existing models, clustering, SHAP logic, and dashboard behavior remain unchanged.

## What it adds

- Graph Neural Network (GNN) embeddings for drug representations
- Embedding-based clustering for new and existing drugs
- Dynamic re-clustering when new drugs are introduced
- Doctor dashboard integration through an optional `handle_doctor_input` API
- Similarity search for drug embeddings

## Architecture

- `gnn/`
  - `features.py`: extracts numeric drug descriptors from SMILES or drug names
  - `graph_builder.py`: builds a drug interaction graph from existing datasets
  - `model.py`: defines a lightweight GNN architecture when PyTorch Geometric is available
  - `embedder.py`: produces and caches embeddings for drugs

- `clustering/`
  - `embedding_cluster.py`: trains embedding-based clusters using KMeans when available
  - `cluster_assigner.py`: assigns new drugs to the nearest embedding cluster

- `similarity/`
  - `similarity_engine.py`: finds the top similar drugs using cosine similarity on embeddings

- `doctor_pipeline/`
  - `doctor_handler.py`: orchestrates doctor input processing and returns structured results
  - `drug_processor.py`: checks new drugs, generates features/embeddings, and persists helpers
  - `interaction_processor.py`: saves doctor-submitted interactions to Excel and updates the graph

- `pipeline.py`: main public entrypoint for the advanced pipeline
- `api_handler.py`: optional adapter for doctor-facing APIs
- `run_pipeline.py`: CLI helper for manual testing

## Optional usage

Import the extension only when needed:

```python
from advanced_ai_pipeline.api_handler import handle_doctor_input

result = handle_doctor_input("Aspirin", "Ibuprofen", "Doctor recommended evaluation")
print(result)
```

## Behavior

The pipeline performs these steps for doctor-submitted pairs:

1. Save the interaction to `data/doctor_added_interactions.xlsx`
2. Load the current graph from base and doctor datasets
3. Extract features from SMILES or drug names
4. Generate or load a drug embedding
5. Assign the drug to an embedding cluster
6. Update graph relationships dynamically
7. Return structured response with clusters and similar drugs

## Notes

- The implementation is designed to be optional and additive.
- If PyTorch Geometric or RDKit are not installed, the pipeline falls back to safe feature-based embeddings.
- Existing application behavior and models are not modified.
