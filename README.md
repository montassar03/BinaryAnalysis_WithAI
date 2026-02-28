# Submission_BinaryAnalysis_WithAI
The evaluation/ directory is organized to clearly separate source code, generated artifacts, logs, and final results to ensure reproducibility and maintainability.
src/
 Contains all Python scripts used for embedding generation and retrieval evaluation (e.g., embedding extraction, ranking, NDCG computation).


embeddings/
 Stores generated embedding vectors (.npy) and corresponding function ID mappings (.fids.txt). These files are model outputs and can be regenerated from the source scripts.


logs/
 Contains execution logs for traceability and debugging.


logs/embed/ – Logs from embedding generation runs


logs/eval/ – Logs from retrieval and evaluation experiments


results/
 Stores final evaluation outputs in JSON format (e.g., retrieval metrics, NDCG scores). These files represent the summarized experimental outcomes.


metadata/
 Includes auxiliary data such as function descriptions and anchor pools used during evaluation.
