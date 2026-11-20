# DGX Spark benchmark notes, November 2026

The numbers below come from the local V6 and Gate7 test sessions and a separate controlled decode test. They record observed runs on one Spark; they are not a cross-machine leaderboard or a statistically significant model evaluation.

## Controlled generation within vLLM 0.34

| Configuration | Decode throughput |
| --- | ---: |
| MTP2, eager execution | 41.69 tokens/s |
| MTP2, CUDA Graphs `FULL_DECODE_ONLY` | 46.55 tokens/s |

The difference is 11.65% in this specific test. Logs from the optimized run reported successful decode graph capture and speculative decoding activity. This comparison does not establish an improvement over vLLM 0.32 under identical conditions.

## Hermes agentic tasks

| Runtime | V6 elapsed | V6 evaluation | Gate7 elapsed | Gate7 hidden evaluation |
| --- | ---: | --- | ---: | --- |
| vLLM 0.32 MAPFIX | 168 s | 24/24 | 677 s | 1 hidden failure |
| vLLM 0.34 MTP2 eager | 290 s | 24/24 | 999 s | 1 hidden failure |
| vLLM 0.34 MTP2 + CUDA Graphs | 196 s | 24/24 | 489 s | 1 hidden failure |

V6 with CUDA Graphs: Hermes returned exit code 0 after 196 seconds, followed by a 24/24 external pytest result. Gate7 with CUDA Graphs: Hermes ran for 489 seconds; 31 visible/generated tests passed, the expanded run passed 45 of 46 tests, and the task-specific checker printed `FIXED_GATE7_PASS`.

The hidden Gate7 failure was `test_serializer_preserves_none`. A pre-existing serializer omitted the `description` key when its value was `None`. The failure was already present in the earlier runtime configurations. V6 had caught a related serializer issue, but Gate7 focused on a new order-quotation feature and left it unchanged.

These runs differ in agent behavior, generated tests and number of tool calls. Gate7 generated a different number of tests in each execution. Reported wall-clock time should not be treated as an isolated CUDA Graphs effect.

## Configuration differences

The 0.34 profile sets one maximum sequence and a 2,048-token batch budget with a fixed 24 GiB KV cache. The local 0.32 MAPFIX configuration used higher concurrency and different memory scheduling. No like-for-like throughput test for eight simultaneous requests or 500,000-token document retrieval is included here.

The original benchmark logs and fixtures remain in the local benchmark workspace and were not included in the source export. This table is a transcription of the recorded local results. Reproducing the agentic results requires publishing the benchmark fixtures, hidden tests and Hermes settings separately.
