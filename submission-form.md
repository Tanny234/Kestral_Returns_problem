# Submission form

**Candidate:** Jyotishka A Badiger  **Date:** 3 Oct 2026

## What did you build?
A model that scores each order's chance of being returned, plus a small web service (POST /score) and a one-page screen that calls it. It returns the probability, a risk band, a recommended action (call before dispatch, or no action), the expected rupee value of a call, and plain-English reasons. Predictions for all 2,096 test orders are in predictions.csv. The model is a 50/50 blend of a small LightGBM and a logistic regression. It runs locally, with no paid API or key.

## Expected score before sending predictions.csv
ROC-AUC about 0.78 (likely 0.74-0.81), PR-AUC about 0.41. Validated by training on Apr 2025-Mar 2026 and testing on Apr-Jun 2026; three other rolling windows gave AUC 0.77-0.79. Accuracy would be about 89-90%, barely above the 88.5% you get by predicting "no return" every time, so accuracy is the wrong measure. The 95% target is not reachable at dispatch.

## How do you know it works, and how often does it not?
Time-based validation (never random), with bootstrap confidence intervals. Calling orders at risk 0.15 or higher flags 484 of 2,126 validation orders and catches 143 of 245 returns. The remaining 102 returns are missed, and many flagged orders would not have been returned. Estimated net gain is Rs 35,777 per quarter (range 27,278-44,543). See EVIDENCE.md.

## What would you push back on?
- The Rs 600 cost of a return: Finance's figure is Rs 1,150, which I used.
- Holding orders: it loses money at every cut-off, so I recommend a call before dispatch instead.
- The 95% accuracy target (see above).

## What is wrong with it?
- The 35% call effect comes from an unverified pilot.
- The 25% margin on cancelled sales is my assumption (I tested 15-35%).
- 889 orders have a default pincode (0), so location tells the model little.
- Unknown customers are scored as new, non-Shield customers, so their risk is less certain.
- The model catches fewer than 6 in 10 returns.

## Cost per prediction
Rs 0. It runs locally with no external API.

## What did you leave out?
- service and pickup columns (last_service_event_type, pickup_scheduled_at) are written after a return, so using them would be leakage. The leaky model hit 99% accuracy and is useless at dispatch.
- Zoho timestamps, which are in UTC and only record resolution events.

## What did you do that nobody asked for?
- Cleaned 651 duplicate partner-feed rows and a x100 order-value error in Oct 2025.
- Rupee analysis of calls vs holds, and a recommended 1-in-4 randomized holdout to measure the true effect of a call.
- DECISIONS.md with 8 written decisions.

## Monday handover
Run README steps, start the service, and begin calling flagged orders at risk 0.15 or higher, keeping a random quarter of them uncalled. After 4-6 weeks compare return rates to replace the 35% assumption with a real number, then revisit the cut-off.

## What is in the pack
predictions.csv (sent through the submission channel, not in the repo), app/ (service + screen), README.md, EVIDENCE.md, DECISIONS.md, MEMO_to_Ritu.md, code to reproduce. Screen recording (about 3:00) submitted separately. It does not show an unknown customer ID; the app handles that case by scoring them as a new customer, as described in the README.

## AI tools: what I used / what it cost / what I discarded
- Used: Claude (chat + coding tools in claude.ai) for data exploration, modelling code, the service and documents. Free plan; cost Rs 0.
- In the product: no model API or paid key. LightGBM + logistic regression run locally, so cost per order is Rs 0.
- Discarded: a leaky model (99% accuracy), an all-feature LightGBM (worse than the simple blend), random cross-validation.
