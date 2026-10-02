# Submission form 

**Candidate:** Jyotishka A Badiger  **Date:** 2 Oct 2026

## Expected score before sending predictions.csv
ROC-AUC ~0.78 (likely 0.74-0.81), PR-AUC ~0.41. Why: validated on Apr-Jun 2026 after training on earlier months only;
three other rolling windows gave AUC 0.77-0.79. Accuracy would be ~89-90%, not 95%; the 95% target is not reachable at dispatch.

## What I decided where the brief was unclear
See DECISIONS.md (8 decisions). Biggest: excluded service/pickup columns as post-outcome leakage; recommended calls, not holds.

## What is in the pack
predictions.csv, app/ (service + screen), README.md, EVIDENCE.md, MEMO.md, RECORDING_SCRIPT.md, code to reproduce.

## AI tools: what I used / what it cost / what I discarded  [CONFIRM - fill in your real figures]
- Used: Claude (chat + coding tools in claude.ai) for data exploration, modelling code, service and documents. Free plan; [your cost: Rs ___].
- In the product: no model API or paid key. The service runs locally with LightGBM + logistic regression, so cost per order is Rs 0.
- Discarded: a leaky model (99% accuracy), an all-feature LightGBM (worse than the simple blend), random cross-validation.

## Known limits
Pilot effect (35%) is unverified; margin on cancelled sales is my assumption (25%); screen recording not included (script provided).
