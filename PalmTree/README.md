Submission_BinaryAnalysis_WithAI
Evaluation Directory Structure
The evaluation/ directory is structured to clearly separate implementation code from generated outputs to keep the project organized and easy to review.

- src/
Contains the Python scripts used for generating embeddings and running retrieval evaluation.
- embeddings/
Stores the generated embedding files (.npy) and their corresponding function ID mappings (.fids.txt).
- logs/
Contains execution logs for embedding generation and evaluation runs. These are included for transparency and debugging purposes.

   - logs/embed/ – Embedding generation logs
   - logs/eval/ – Evaluation logs
- results/
Contains the final evaluation outputs (e.g., retrieval metrics and NDCG scores in JSON format).
- metadata/
Includes auxiliary input files required during evaluation (e.g., function descriptions or anchor pools).

This structure ensures a clear separation between source code and generated outputs, making the project easier to understand, navigate, and assess.
