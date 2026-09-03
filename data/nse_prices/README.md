# NSE bhavcopy price store

Rolling ~15-month window of exchange-official daily prices from the NSE equity
bhavcopy, maintained by [`scripts/nse_price_history.py`](../../scripts/nse_price_history.py)
and consumed by the Nifty 50 / Nifty 500 rotation scripts (`--price-source bhavcopy`,
the default).

Why this exists: Yahoo Finance is unreliable from GitHub-hosted runners — its
multi-ticker endpoint silently drops a large share of NSE symbols, which starved
the Nifty 500 rotation of coverage and stalled the dashboard. One bhavcopy file
per session gives ~100% coverage.

## Files

| File | Contents |
|---|---|
| `YYYY-MM.csv.gz` | `d,symbol,open,high,low,close,prev_close,volume` for that month. Index levels are stored under the pseudo-symbols `__NIFTY50__` / `__NIFTY500__`. Only the current month's file changes on a normal refresh. |
| `corp_actions.json` | `{SYMBOL: [[ex_date, price_factor], ...]}` — split/bonus/demerger factors used to build the adjusted `Adj Close`. Ratios come from the NSE corporate-actions API and are confirmed against the actual price step. |

## Refresh

```
python scripts/nse_price_history.py --data-dir data/nse_prices --lookback-days 430
```

Fetches only the sessions missing from the store, rewrites the affected monthly
file(s), refreshes `corp_actions.json`, and trims months that fell out of the
window. Safe to run repeatedly; resilient to the NSE corporate-actions API being
temporarily unreachable (last-known ratios are kept).
