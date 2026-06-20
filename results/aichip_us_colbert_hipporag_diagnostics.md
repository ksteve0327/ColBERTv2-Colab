# AIChip-US HippoRAG ColBERTv2 Trigger/IRCoT Diagnostics

## 1. Trigger linking 의심 사례

### Single-step HippoRAG (ColBERTv2)

| id | type | R@5 | suspicious links | worst linked examples |
| --- | --- | --- | --- | --- |
| global_007 | global | 1.0 | 1/1 | die integration patents → inter die communication (0.4864, overlap 0.333) |
| local_008 | local | 0.0 | 3/5 | multiplexed register rotation → register file (0.3896, overlap 0.333)<br>512 processing units → processing neural networks (0.4625, overlap 0.333)<br>registers receiving memory rows → memory cell array (0.4691, overlap 0.25) |
| local_009 | local | 1.0 | 1/4 | high bandwidth  high capacity memory → high electric field cell (0.4452, overlap 0.25) |
| global_008 | global | 0.5 | 1/4 | high bandwidth memory → hierarchical memory (0.4658, overlap 0.333) |
| global_004 | global | 0.0 | 1/4 | cooperative ann computation → neural network computation (0.4626, overlap 0.333) |
| local_002 | local | 1.0 | 1/4 | output tensor values → output data (0.4968, overlap 0.333) |
| local_003 | local | 1.0 | 1/5 | virtual channel specifier → channel formation region (0.4176, overlap 0.333) |
| local_001 | local | 1.0 | 1/6 | data layout transformations → data structures (0.4893, overlap 0.333) |

### IRCoT + HippoRAG (ColBERTv2) step-2 thought query

| id | type | R@5 | suspicious links | worst linked examples |
| --- | --- | --- | --- | --- |
| global_004 | global | 0.5 | 2/5 | high bandwidth access → high electric field cell (0.4459, overlap 0.25)<br>multiple processing units → processing neural networks (0.4584, overlap 0.333) |
| local_008 | local | 1.0 | 2/6 | multiplexed register rotation → register file (0.3896, overlap 0.333)<br>registers → register (1.0, overlap 0.0) |
| global_006 | global | 0.5 | 2/7 | off chip memory → memory cells (0.4727, overlap 0.333)<br>data layout → data stream assembly control (0.4888, overlap 0.25) |
| global_010 | global | 0.6666666666666666 | 1/4 | data layout → data stream assembly control (0.4888, overlap 0.25) |
| local_003 | local | 1.0 | 2/8 | data element specifier → data associated with operation (0.3957, overlap 0.333)<br>virtual channel specifier → channel formation region (0.4176, overlap 0.333) |
| local_009 | local | 1.0 | 1/5 | high bandwidth memory → hierarchical memory (0.4658, overlap 0.333) |
| local_002 | local | 1.0 | 1/5 | output tensor value statistics → output data (0.4945, overlap 0.25) |
| global_001 | global | 1.0 | 1/7 | data layout → data stream assembly control (0.4888, overlap 0.25) |

## 2. Query NER / fallback 점검

- 원본 QA query 20개 중 NER cache 누락: 0개 `[]`
- n-gram/stopword성 entity가 있는 원본 QA query: 9개 `['global_002', 'global_003', 'global_004', 'global_005', 'global_006', 'global_007', 'global_008', 'global_010', 'local_001']`
- IRCoT HippoRAG step-2 thought query는 `aichip_us_colbert_ircot_thoughts.json`의 `named_entities`를 사용한 것으로 보이며, 해당 id는 20개다. 따라서 thought fallback을 전부 제외하면 IRCoT+HippoRAG step-2 평가 subset이 사실상 비게 된다.

### Poor original NER 제거 후 재집계

| method | subset | local R@5 | global R@5 | avg R@5 | n |
| --- | --- | --- | --- | --- | --- |
| ColBERTv2 | single clean-NER | 100.0 | 58.3 | 92.4 | 11 |
| HippoRAG (ColBERTv2) | single clean-NER | 88.9 | 66.7 | 84.8 | 11 |
| IRCoT + ColBERTv2 | IRCoT clean-original-NER | 100.0 | 83.3 | 97.0 | 11 |
| IRCoT + HippoRAG (ColBERTv2) | IRCoT clean-original-NER | 100.0 | 66.7 | 93.9 | 11 |

## 3. Global-only 확인

| method | setting | global R@2 | global R@5 |
| --- | --- | --- | --- |
| ColBERTv2 | single | 41.7 | 55.0 |
| HippoRAG (ColBERTv2) | single | 16.7 | 50.0 |
| IRCoT + ColBERTv2 | IRCoT | 51.7 | 78.3 |
| IRCoT + HippoRAG (ColBERTv2) | IRCoT | 31.7 | 60.0 |

결론: 현재 결과에서는 global-only로 봐도 HippoRAG 우위가 선명해지지 않는다. 오히려 ColBERTv2 단독 및 IRCoT+ColBERTv2가 더 높다. 가장 직접적인 원인은 query entity가 KG phrase node에 부정확하게 연결되는 trigger linking 품질로 보인다.
