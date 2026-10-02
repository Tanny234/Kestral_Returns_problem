# Decisions I made (nobody to ask)
1. Dropped last_service_event_type and pickup_scheduled_at: written after the return; train and test differ (test has no pickups).
2. De-duplicated partner-feed rows (same label every time) before validating.
3. Divided Oct-2025 order values by 100 (value/(list price x qty x discount) was exactly 100). Model uses that ratio.
4. Validated by time (train to Mar 2026, test Apr-Jun 2026), not random folds.
5. Return cost Rs 1,150 (Finance, policy) not Rs 600 (Ritu). Hold margin loss is my assumption: 25% of order value, shown 15-35%.
6. Assumed every call completes and the pilot's 35% reduction holds; flagged as unverified.
7. Cut-off 0.15, just above the 0.11 break-even, for a safety margin.
8. No LLM or paid API in the product; reasons come from the logistic part of the model.
