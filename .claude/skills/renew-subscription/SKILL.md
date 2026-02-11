---
name: renew-subscription
description: Renew the current subscription by updating data/settings.json with new dates and exchange rate
disable-model-invocation: true
allowed-tools: Read, Edit, WebFetch, WebSearch, AskUserQuestion, Bash(date *)
---

# Renew Subscription

Renew the current Claude subscription in `data/settings.json`.

## Steps

1. **Read current settings** from `data/settings.json` in the project root. Identify the latest subscription entry (highest `end` date).

2. **Compute new dates.** The new subscription starts the day after the previous one ends. The new end date is one calendar month later minus one day (e.g. start Mar 1 → end Mar 31, start Jan 29 → end Feb 28). Use Python date logic if needed.

3. **Fetch the current exchange rate.** The current subscription has a `currency` field (e.g. "GBP") and a `usd_rate` field (USD per 1 unit of currency). Use a free exchange rate API to look up the current rate:
   - Try: `https://open.er-api.com/v6/latest/USD` — read the `rates.<CURRENCY>` field, then compute `usd_rate = 1 / rates.<CURRENCY>` since we need USD-per-1-unit (e.g. if rates.GBP = 0.73, then usd_rate = 1/0.73 ≈ 1.3699).
   - If the API is unavailable, fall back to WebSearch for "USD to <CURRENCY> exchange rate" and extract the rate.

4. **Ask user to confirm** all details before writing. Show them:
   - Plan name (default: same as previous)
   - Cost and currency (default: same as previous)
   - New start and end dates
   - Fetched USD exchange rate (rounded to 5 decimal places)
   - The old rate for comparison

   Use AskUserQuestion to let the user confirm or provide corrections. If they want to change anything, incorporate their feedback.

5. **Append the new subscription** to the `subscriptions` array in `data/settings.json` using the Edit tool. Do NOT remove or modify existing subscriptions. Ensure valid JSON formatting.

6. **Verify** by reading the file back and confirming it parses as valid JSON with no overlapping subscriptions.
