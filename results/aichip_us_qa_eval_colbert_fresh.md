# Patent QA Answer Performance

QA set: `hipporag_v1/data/aichip_us_qa_dev.json`

Table 4-style QA performance on the AIChip-US patent QA dev set.

| Retriever | Local EM | Local F1 | Global EM | Global F1 | Average EM | Average F1 |
|---|---:|---:|---:|---:|---:|---:|
| ColBERTv2 | 0.0 | 76.0 | 0.0 | 29.2 | 0.0 | 52.6 |
| HippoRAG (ColBERTv2) | 0.0 | 57.9 | 0.0 | 33.9 | 0.0 | 45.9 |
| IRCoT + ColBERTv2 | 0.0 | 76.3 | 0.0 | 41.3 | 0.0 | 58.8 |
| IRCoT + HippoRAG (ColBERTv2) | 0.0 | 74.6 | 0.0 | 30.5 | 0.0 | 52.6 |
