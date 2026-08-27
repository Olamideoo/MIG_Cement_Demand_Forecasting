# Cement Demand Forecasting — Final Report

**For:** MIG Operations
**Scope:** 30 sites, weekly cement demand, 8 weeks ahead
**Status:** Deployed and running — <http://13.59.81.166/>

---

## 1. In one paragraph

Across the eight weeks we tested, MIG's sites were ready to pour on **45% of the days
a pour was planned**. The estate was not short of cement — it ordered 42,549 tonnes
in that window and rejected 8,755 tonnes of it at the silo because there was no room.
Cement was in the wrong place at the wrong time. We built a weekly demand forecast per
site and used it to drive reordering. Simulated over the same eight weeks, it raises
pour readiness to **99.9%** while ordering **roughly a fifth less cement**, and reduces
unmet pour from 10,684 tonnes to under 20.

---

## 2. What we found in the current way of working

Three things stood out, all measured over the same eight-week window used to score
everything else in this report.

**Silos were either jammed or starved 68% of the time.** Only 18% of site-days sat in a
workable 20–80% fill band. The estate was hoarding and starving simultaneously — some
sites so full that deliveries bounced, others empty when the mixer arrived.

**One tonne in five that was delivered never made it into a silo.** 8,755 tonnes —
20.6% of deliveries — were rejected on arrival for lack of headroom. That is haulage
paid for, and a delivery slot used, for cement that went straight back out.

**Ordering was driven by the pour schedule, not by what sites actually consume.**
39.7% of recorded consumption is stock-censored — the site consumed what it had, not
what it needed — so the schedule systematically overstates real demand at exactly the
sites that are already short.

The pattern is not a shortage of cement. It is a distribution and timing problem, and
that is the sort of problem a forecast can fix.

---

## 3. What changes

The system forecasts each site's weekly consumption eight weeks ahead, then runs a
reorder policy against those forecasts: when projected stock falls to the reorder point,
order back up to target. Reorder points are set per site from that site's own forecast
error, so a predictable site carries less buffer than a volatile one.

Simulated over the same eight weeks, against recorded practice:

| | Recorded practice | Forecast-driven policy |
|---|---:|---:|
| Pour readiness | 45% | **99.9%** |
| Cement ordered | 42,549 t | **32,924 t** |
| Unmet pour | 10,684 t | **18 t** |
| Deliveries rejected at silo | 8,755 t | **0 t** |
| Site-days jammed or starved | 68% | **8%** |

Less cement, ordered at better times, with sites almost always able to pour. The
reduction in ordering is not a saving squeezed out of service — service improves at the
same time, because the cement that was previously rejected or sitting in the wrong silo
is no longer bought.

**Where it lands day to day.** Of the 30 sites, the current run flags **7 red** (will
stock out within lead time), **7 amber** (below reorder point) and **16 green**. Total
suggested order across the flagged sites is about 3,000 tonnes. Reorder points range
from 7 to 232 tonnes depending on the site's size and volatility.

---

## 4. How accurate the forecast is

Measured on eight weeks the model had never seen, after training on everything up to
30 September 2024:

- **12.8% mean absolute percentage error** across the estate, against a 15% target
- **+1.2% net bias** — it over-forecasts very slightly, which is the safer direction
- Total forecast 39,345 t against 38,884 t actual

Two things matter operationally more than the headline.

**Accuracy does not decay with horizon.** Week 1 error is 14.6%; week 8 is 9.8%. That is
unusual, and it happens because the forecast leans on the planned pour schedule, which
is known weeks ahead. **An eight-week forecast is as trustworthy as a one-week
forecast** — so order horizons can be set by haulage economics rather than by how far
ahead the model stays sensible.

**Per-site accuracy varies widely.** The estate meets the 15% target, but individual
sites range from 4.2% to 22.6%, and **18 of 30 sites are individually within 15%.** The
12 that are not are still usable — the reorder policy already gives them larger buffers,
because buffers are set from each site's own error — but they warrant more attention
from the planner, and they are the natural first target for improvement.

---

## 5. How to use it safely

**Do not ask it about pours the site has never done.** The model learns from what each
site has actually poured. Ask it to price a 300-tonne week at a site whose largest ever
week is 118 tonnes and it will return a number, but that number is borrowed from other
sites, not a forecast for this one. The `/predict` endpoint flags this explicitly and
returns the site's real pour range alongside the answer — check that flag before acting
on a what-if.

**Treat the reorder flags as a shortlist, not an instruction.** They encode a 3-day lead
time and a ~98% service level. Both are our assumptions, not facts from the data
(section 7). A site with a known haulage constraint needs a planner's judgement over the
default.

**The intervals mean something.** Each forecast comes with a range built from that
site's own historical error. A wide range is the model saying it does not know this site
well. Treat a wide range at a red site as a reason to order early.

---

## 6. What the system does not know

- **Weather beyond what is already in the schedule.** Rainfall above roughly 25 mm/day
  raises no-pour days from 9% to 68%, but the model does not receive a weather forecast.
  A wet week will look like an unexplained over-forecast.
- **Anything about a site's first weeks.** New sites have no history to learn from.
- **Cost.** No price, haulage or holding-cost data exists in the source system, so every
  figure in this report is in tonnes and days. Converting the ~9,600-tonne ordering
  reduction into money needs MIG's own cost assumptions.
- **Cement grade.** All three grades are forecast together as total site demand. The
  data does not support a reliable split — the stock ledger only balances at site level.

---

## 7. What we need MIG to confirm

Five judgements were made without a definitive answer in the data. Each is recorded in
our decisions log with its reasoning, and each is currently marked provisional pending
MIG's confirmation.

1. **Delivery lead time of 3 days.** Not present in the source data. Every reorder point
   depends on it.
2. **Service level of ~98%.** Sets how much safety stock each site carries. Higher means
   more cement standing idle; lower means more stockouts.
3. **Ledger repair method.** Delivery records exceed silo headroom in places. We cap
   deliveries at available headroom, which rejects 19.2% of recorded delivered volume
   but moves total consumption by only −0.49%. If the true explanation is
   mis-recorded capacity rather than over-delivery, this needs revisiting.
4. **Forecasting consumption rather than planned pour.** Because 39.7% of consumption is
   stock-censored, we forecast what sites actually used. If MIG considers the pour
   schedule the operational truth, the target changes.
5. **Weekly rather than daily forecasts.** The 15% accuracy target is not reachable at
   daily grain on this data. Weekly is what makes the target achievable.

---

## 8. Where it runs

| | |
|---|---|
| Operations dashboard | <http://13.59.81.166/> |
| Forecast API | <http://13.59.81.166/api/docs> |
| Model tracking | <http://13.59.81.166/mlflow/> |

The dashboard is the day-to-day view: estate overview, per-site drill-down, and the
reorder alert list. The API serves the same numbers to any other system that needs them.
Model tracking records every training run, so a change in behaviour can always be traced
to a specific model.

Retraining is a single command and takes about ninety seconds. We recommend retraining
monthly, or whenever a new site comes online.

---

## Appendix — model at a glance

| | |
|---|---|
| Model | Random forest, 300 trees |
| Grain | Weekly per site, weeks labelled by Monday |
| Trained on | 3 Jan 2022 – 30 Sep 2024, 4,320 site-weeks, 30 sites |
| Tested on | 7 Oct – 25 Nov 2024, 240 site-weeks, never seen in training |
| Inputs | Planned pour, 7- and 14-day forward pour, days since last pour, silo capacity, site, region, behaviour profile |
| Hold-out MAPE | 12.77% (target ≤ 15%) |
| Hold-out RMSE | 30.57 t per site-week |
| Hold-out bias | +1.92 t per site-week |
| Sites within 15% | 18 of 30 |

**A note on what "tested" means.** The eight test weeks are weeks that had already
happened when we built this — that is how we can compare forecast against actual and
state the error. To forecast genuinely future weeks, the system needs the pour schedule
for those weeks, which MIG already produces. Nothing further is required, but the
accuracy quoted here is measured on history, not on live operation.
