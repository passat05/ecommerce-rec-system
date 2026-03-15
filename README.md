# E-commerce Recommendation System

End-to-end production-ready recommendation system for the **Retailrocket** e-commerce dataset.  
Uses **Two-Tower Neural Retrieval** + **FAISS ANN Search** + **DeepFM Ranking**.  
Pipeline: DuckDB preprocessing → Two-Tower Neural Retrieval → FAISS ANN Search → DeepFM + LambdaRank Ranking → FastAPI serving.

---

## Architecture

```
Offline Pipeline                               Online Pipeline
─────────────────────────────────────          ──────────────────────────────────
Retailrocket CSVs                              POST /recommend {user_id}
     │                                                │
  preprocess.py                               memmap user embedding lookup
  (chunk props, encode, split)                       │
     │                                        FAISS retrieve top-50 items
  train_two_tower.py                                 │
  (InfoNCE, user history pooling)             DeepFM re-rank (LambdaRank)
     │                                               │
  export_embeddings.py                        Return top-K as JSON
     │                                               
  build_index.py          
     │
  train_ranker.py
  (user emb memmap + DeepFM training)
     │
  evaluate.py
```

---

## Model Design

### Two-Tower Retrieval
| Component | Detail |
|-----------|--------|
| User Tower | Weighted average of last-K item embeddings → MLP → L2-norm |
| Item Tower | item_emb + category_emb + price_emb → MLP → L2-norm |
| Training   | InfoNCE loss with in-batch negatives (batch=2048) |
| Output     | 64-dim L2-normalised vectors, dot-product similarity |

### DeepFM Ranking
| Component | Detail |
|-----------|--------|
| Fields    | 7 fields: category_id, event_type, price_bucket, recency_bucket, popularity_bucket, user_emb_proj, item_emb_proj |
| FM part   | Efficient order-2 feature interactions |
| Deep part | MLP 256→128→64→1 with LayerNorm + Dropout |
| Output    | Sigmoid score for ranking |

---

## Project Structure

```
ecommerce-rec-system/
│
├── data/
│   ├── raw/                         # Retailrocket CSVs
│   ├── processed/                   # Auto-generated artefacts
│   |__ preprocess.py
│
├── retrieval/
│   ├── two_tower.py                 # TwoTowerModel + InfoNCE loss
│   ├── dataset.py                   # TwoTowerDataset
│   ├── train_two_tower.py           # Training + item embedding export
│   └── export_embeddings.py         # L2 normalisation utility
│
├── ranking/
│   ├── deepfm_model.py             # DeepFM (FM + Deep MLP)
│   ├── dataset.py                  # RankingDataset with memmap user emb
│   |__ train_ranker.py             # User emb generation + DeepFM training
│
├── faiss_index/
│   ├── build_index.py
│   └── search.py                   # FAISSRetriever singleton
│
├── api/
│   ├── main.py                     # FastAPI with MPS/CPU device + memmap
│   └── schema.py
│
├── evaluation/
│   ├── metrics.py                  # Recall@K, NDCG@K, Precision@K, MRR
│   └── evaluate.py                 # Exp1: retrieval | Exp2: +DeepFM
│
├── configs/config.yaml
├── scripts/
│   ├── train_pipeline.py           # Full offline orchestrator
│   └── inference_pipeline.py       # Batch inference smoke-test
│
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### 1. Environment Setup

```bash
source .venv/bin/activate
```

### 2. Train

```bash
# Full pipeline (recommended)
python scripts/train_pipeline.py

# Skip preprocessing if data already processed
python scripts/train_pipeline.py --skip-preprocess

# Skip retrieval if Two-Tower already trained
python scripts/train_pipeline.py --skip-retrieval
```

Pipeline steps:
| Step | Module | Key output |
|------|--------|------------|
| 1 | `data.preprocess` | `train/val/test.parquet`, `item_meta.parquet`, ID maps |
| 2 | `retrieval.train_two_tower` | `item_embeddings.npy`, `two_tower.pt` |
| 3 | `retrieval.export_embeddings` | float32 normalised embeddings |
| 4 | `faiss_index.build_index` | `item_index.faiss` |
| 5 | `ranking.train_ranker` | `user_embeddings.npy` (memmap), `deepfm.pt` |
| 6 | `evaluation.evaluate` | `evaluation/results/evaluation_results.json` |

### 3. Serve

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test

```bash
# Health
curl http://localhost:8000/health

# Recommend for user 42
curl -X POST http://localhost:8000/recommend \
     -H "Content-Type: application/json" \
     -d '{"user_id": 123, "top_k": 5}'
```

**Response:**
```json
{
   "user_id":123,
   "recommendations":[
      {"item_id":46154,"original_id":"91264","score":0.9957},
      {"item_id":13633,"original_id":"27127","score":0.9952},
      {"item_id":231467,"original_id":"459631","score":0.9942},
      {"item_id":35672,"original_id":"70527","score":0.9915},
      {"item_id":113976,"original_id":"226378","score":0.9702}
   ]
}
```

### 5. Batch Inference

```bash
python scripts/inference_pipeline.py --user-ids 0 1 2 3 --top-k 5
python scripts/inference_pipeline.py --user-ids 0 1 --output recs.json
```

---

## Docker Deployment

```bash
# Build
docker build -t ecommerce-rec .

# Run with pre-trained artefacts mounted
docker run -p 8000:8000 \
  -v $(pwd)/data/processed:/app/data/processed \
  -v $(pwd)/faiss_index:/app/faiss_index \
  -v $(pwd)/ranking/deepfm.pt:/app/ranking/deepfm.pt \
  -v $(pwd)/retrieval/two_tower.pt:/app/retrieval/two_tower.pt \
  ecommerce-rec
```

---

## Configuration

All hyperparameters in `configs/config.yaml`:

| Section | Key | Default | Notes |
|---------|-----|---------|-------|
| `retrieval` | `embedding_dim` | 64 | Two-Tower output dim |
| `retrieval` | `batch_size` | 2048 | Safe for 16 GB Mac |
| `retrieval` | `temperature` | 0.07 | InfoNCE temperature |
| `ranking` | `field_emb_dim` | 16 | DeepFM field embedding dim |
| `ranking` | `hidden_dims` | [256,128,64] | Deep MLP layers |
| `faiss` | `top_n` | 50 | Retrieval candidates |
| `data` | `history_len` | 20 | Last-K items for user tower |
| `data` | `props_chunk_size` | 200000 | Chunk size for item properties |

---

## Evaluation

```bash
python -m evaluation.evaluate
```

Results printed and saved to `evaluation/results/evaluation_results.json`:

```
================================================================
Metric                    Two-Tower+FAISS             + DeepFM
================================================================
Recall@5                           0.0551             0.0683 (+0.0132)
NDCG@5                             0.0443             0.0496 (+0.0053)
Recall@10                          0.0697             0.0865 (+0.0168)
NDCG@10                            0.0498             0.0559 (+0.0061)
Recall@20                          0.0845             0.0986 (+0.0141)
NDCG@20                            0.0540             0.0596 (+0.0055)
================================================================
```
