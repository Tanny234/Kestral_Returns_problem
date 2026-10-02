# Evidence: what works, and how often it does not

## 1. The 95% target
Only 11.4% of orders are returned, so "predict no return for everything" is already 88.5% accurate.
The best honest model reaches 89.6% accuracy on the last 3 months of training data. 95% is not reachable from
information available at dispatch. A model that appears to reach 99% (AUC 0.999, 99.25% accuracy) does so by reading
`last_service_event_type` and `pickup_scheduled_at`, which are written only after a customer has already returned the
item (REVERSE_PICKUP rows are 100% returned; DEMO_DONE / INSTALL_DONE rows are 0% returned). The test file contains none
of these, so that model would fail in production. I excluded both columns.

## 2. Validation design
Train on Apr 2025 - Mar 2026, test on Apr - Jun 2026 (2,126 orders, 245 returns), the same "future" shape as the
Jul - Sep 2026 test file. Three further rolling windows gave the same picture.
Duplicates (651 partner-feed order ids, each twice, same label) were removed first; otherwise validation rows can be
copies of training rows.

| Measure (Apr-Jun 2026) | Result | 95% interval |
|---|---|---|
| ROC-AUC | 0.781 | 0.748 - 0.811 |
| PR-AUC (base rate 0.115) | 0.413 | 0.348 - 0.476 |
| Rolling-window AUC | 0.776 - 0.786 | |
| Calibration | predicted 4% -> actual 3%; 12% -> 12%; 31% -> 32% | |

Models compared: LightGBM with all features (AUC 0.764), LightGBM with fewer features (0.770 - 0.781),
logistic regression (0.776 - 0.787), 50/50 blend (0.776 - 0.786, steadier). I used the blend.

## 3. How often it is wrong (flagging the top 23% of orders, score >= 0.15)
- Flagged 484 of 2,126 orders. 143 were real returns (precision 35%); 341 flagged orders were fine.
- 143 of the 245 returns were caught (recall 58%); 102 returns were missed.
- So roughly 2 of 3 flagged orders are false alarms, and 4 in 10 returns slip through. This is a ranking tool, not an oracle.

## 4. The rupees (same 3 months)
Costs from ops-policy: return Rs 1,150; call Rs 45; call prevents 35% of returns on called orders; hold -> 12% cancel.

| Action | Net per quarter |
|---|---|
| Do nothing | returns cost Rs 2.82 lakh |
| Call everyone | +Rs 2.9k |
| Call orders scoring >= 0.15 | +Rs 35.8k (27k - 45k) |
| Hold orders scoring >= 0.15 | -Rs 62k (assuming 25% margin; -31k at 15%, -94k at 35%) |

Holding loses money at every cut-off I tried: a hold only prevents a return if the customer cancels (12%), and
cancellations hit good orders as often as bad ones. Break-even for a call is a 11% return chance (Rs 45 / (0.35 x Rs 1,150)).
Sensitivity: if calls prevent only 20% of returns (not 35%), the net falls to +Rs 11k; at 15% it is ~zero.
Cap on benefit: even a perfect model with free calls saves at most 35% of Rs 2.82 lakh = Rs 99k per quarter.

## 5. Other findings
- Oct 2025 order values were stored x100 (new gateway); corrected before use. Test file is clean (Jul - Sep 2026).
- 889 walk-in orders have pincode 000000; flagged as "no address" (no effect on returns).
- Shield members return at 18.7% vs 9.3%, so they make up 44% of the called orders, but a call is not a hold.
- No drift between late training months and the test file in discount, delivery promise, payment mix or Shield share.
- The pilot's 35% effect is not a randomised trial; it should be verified (see memo).
- Not used: the legacy Zoho UTC timestamp issue affects resolution events only, which I excluded as leakage anyway.

## 6. Expected score on the hidden outcomes
ROC-AUC about 0.78 (likely range 0.74 - 0.81); PR-AUC about 0.41; accuracy not applicable to scores, and at best ~89-90%
if forced to a yes/no. Mean predicted score on test 0.118 vs train return rate 0.114.
