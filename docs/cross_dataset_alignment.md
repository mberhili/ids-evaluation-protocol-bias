# Semantic Feature Alignment — Cross-Dataset (Response to M5)

## Why this exists

The original submission aligned features across datasets by simple index-based truncation: for a (source, target) pair with feature counts $n_s$ and $n_t$, both feature matrices were trimmed to the first $\min(n_s, n_t)$ columns. Reviewer #2 (M5) correctly observed that this conflates two effects: the inability of models to generalize across datasets (the phenomenon we want to measure) and the artificial mismatch introduced when columns at the same index represent semantically different network properties.

This document describes the semantic alignment we use in the revised cross-dataset experiments.

## Pivot B design

We use **UNSW-NB15** as the semantic pivot. Among the three benchmarks, UNSW-NB15 is the only one that contains *both* flow-based features (compatible with CIC-IDS2017's CICFlowMeter representation) *and* connection-counting / categorical features (compatible with NSL-KDD's DARPA-derived features). It is therefore the only valid pivot in this benchmark set.

Two pairs are reported:

- **UNSW-NB15 ↔ CIC-IDS2017** — 14 semantically aligned features (flow paradigm match).
- **UNSW-NB15 ↔ NSL-KDD** — 10 semantically aligned features (mixed paradigm match).

The third pair, **NSL-KDD ↔ CIC-IDS2017**, is intentionally not reported. Direct semantic alignment between NSL-KDD's content-based features (`hot`, `num_failed_logins`, `is_guest_login`, etc.) and CIC-IDS2017's CICFlowMeter flow statistics yields only three strictly equivalent features (`duration`, `src_bytes`, `dst_bytes`). The manuscript discusses this paradigm incompatibility as itself a finding about benchmark non-comparability.

## Mapping tables

### UNSW-NB15 ↔ CIC-IDS2017 (14 features, flow paradigm)

| #  | UNSW-NB15           | CIC-IDS2017                      | Match type     | Notes                                     |
|----|---------------------|----------------------------------|----------------|-------------------------------------------|
| 1  | `dur`               | `Flow Duration`                  | strict         | Sec vs μs; rescaled by MinMaxScaler.      |
| 2  | `spkts`             | `Total Fwd Packets`              | strict         | Forward-direction packet count.            |
| 3  | `dpkts`             | `Total Backward Packets`         | strict         | Backward-direction packet count.           |
| 4  | `sbytes`            | `Total Length of Fwd Packets`    | strict         | Forward-direction byte count.              |
| 5  | `dbytes`            | `Total Length of Bwd Packets`    | strict         | Backward-direction byte count.             |
| 6  | `rate`              | `Flow Packets/s`                 | strict         | Per UNSW-NB15 documentation, `rate` is pps. |
| 7  | `sinpkt`            | `Fwd IAT Mean`                   | strict         | Mean fwd inter-arrival time.               |
| 8  | `dinpkt`            | `Bwd IAT Mean`                   | strict         | Mean bwd inter-arrival time.               |
| 9  | `sjit`              | `Fwd IAT Std`                    | quasi-strict   | Jitter ≈ std(IAT); definitionally close.   |
| 10 | `djit`              | `Bwd IAT Std`                    | quasi-strict   | Same.                                      |
| 11 | `swin`              | `Init_Win_bytes_forward`         | strict         | Initial TCP window, fwd.                   |
| 12 | `dwin`              | `Init_Win_bytes_backward`        | strict         | Initial TCP window, bwd.                   |
| 13 | `smean`             | `Fwd Packet Length Mean`         | strict         | Mean fwd packet size.                      |
| 14 | `dmean`             | `Bwd Packet Length Mean`         | strict         | Mean bwd packet size.                      |

### UNSW-NB15 ↔ NSL-KDD (10 features, mixed paradigm)

| #  | UNSW-NB15           | NSL-KDD                            | Match type            | Notes                                                                |
|----|---------------------|------------------------------------|------------------------|----------------------------------------------------------------------|
| 1  | `dur`               | `duration`                         | strict                 | Both in seconds.                                                     |
| 2  | `proto`             | `protocol_type`                    | strict-categorical     | Unified vocabulary {tcp, udp, icmp, other}.                          |
| 3  | `service`           | `service`                          | strict-categorical     | Vocabularies overlap but differ (UNSW: 13, NSL: ~70). Per-dataset encoding. |
| 4  | `state`             | `flag`                             | strict-categorical     | Connection state. Vocabularies differ semantically.                  |
| 5  | `sbytes`            | `src_bytes`                        | strict                 | Source-to-destination bytes.                                         |
| 6  | `dbytes`            | `dst_bytes`                        | strict                 | Destination-to-source bytes.                                         |
| 7  | `is_sm_ips_ports`   | `land`                             | strict                 | Land-attack flag: src_ip=dst_ip and src_port=dst_port.               |
| 8  | `ct_dst_ltm`        | `dst_host_count`                   | approximate            | Both: connections to same destination over a window.                 |
| 9  | `ct_srv_dst`        | `dst_host_srv_count`               | approximate            | Both: connections sharing service+destination.                       |
| 10 | `ct_src_dport_ltm`  | `dst_host_same_src_port_rate`     | approximate            | UNSW count vs. NSL rate; semantically related.                       |

## Categorical encoding

For UNSW-NB15 ↔ NSL-KDD, three of the ten features are categorical. Encoding strategy:

- **`proto` / `protocol_type`**: unified mapping `{tcp:0, udp:1, icmp:2, other:3}` is applied identically to both datasets. This is the only categorical for which a coherent cross-dataset code makes sense.
- **`service`**: per-dataset `LabelEncoder`. UNSW-NB15 has 13 service values; NSL-KDD has ~70. The vocabularies overlap but differ; forcing a unified code would either drop most NSL-KDD values or assign large, sparsely populated codes. We accept that the encoded values differ between datasets, which is itself a contributor to the cross-dataset transfer gap (and is consistent with realistic deployment, where service taxonomies differ between organisations).
- **`state` / `flag`**: per-dataset `LabelEncoder`. The TCP-state taxonomies of UNSW-NB15 and NSL-KDD are different ({FIN, INT, CON, ECO, REQ, RST, ...} vs. {SF, S0, S1, S2, S3, REJ, RSTO, RSTR, RSTOS0, OTH, SH}). Same rationale.

After encoding, `MinMaxScaler` is fit on the source training set and applied to all source/target splits, in line with the rest of the pipeline.

## Implementation

The mapping is defined in `scripts/feature_alignment.py` as the constants `UNSW_CIC_MAPPING` and `UNSW_NSL_MAPPING`. The function `load_aligned_pair(pair)` returns the four splits (source train/test, target train/test) with the aligned feature subset, encoded and scaled. Cross-dataset experiments are driven from `scripts/run_cross_dataset_aligned.py`.

## Limitations

The semantic alignment is a best-effort manual mapping. It does not address:

1. **Distributional shift after alignment** — even semantically equivalent features have different distributions across datasets, which standard scaling does not correct.
2. **Sub-feature granularity** — e.g., NSL-KDD `count` uses a 2-second window in some configurations; UNSW-NB15 `ct_dst_ltm` uses a 100-connection window.
3. **Categorical vocabulary mismatches** — encoded as per-dataset, deliberately.

A more principled treatment using domain-adaptation techniques (autoencoder bottlenecks, schema-agnostic foundation models for tabular data) is possible future work and noted in Section 6 of the manuscript.
