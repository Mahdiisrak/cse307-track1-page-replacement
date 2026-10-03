# 3–5 Minute Walkthrough

## 0:00–0:30 — Problem

“FIFO and LRU make fixed assumptions about memory access behavior. This project tests what happens when a locality-heavy workload changes into a random/bursty workload.”

## 0:30–1:30 — Classical policies

Open `track1_page_replacement.py` and show `simulate()`. Explain:

- FIFO evicts the page that entered first.
- LRU evicts the page that has not been used for the longest time.
- Optimal evicts the page whose next use is farthest in the future; it is an offline benchmark.

## 1:30–2:15 — Learned policy

Show `train_learned_selector()`. The decision tree is trained on a separate trace. Its features are current-state features only: age, frequency, recency rank, and page number. The target is the offline best victim from the training trace.

## 2:15–3:00 — Workload shift

Show `make_trace()`. The first half favors pages 0–4 and occasional sequential references. The second half introduces bursts and random references over pages 0–9.

## 3:00–4:00 — Results

Open `results/hit_ratio_by_phase.png` and `results/faults_by_phase.png`. Compare the two phases rather than only the overall average. Mention that Optimal is an upper-bound reference because it can see the future.

## 4:00–5:00 — Conclusion

Explain which policy loses the most hit ratio after the shift, whether the learned policy follows the new behavior, and the main limitation: a small synthetic trace and a shallow classifier cannot represent every real workload.
