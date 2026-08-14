# Options Research Harness

A backtesting / strategy-development harness for defined-risk options
strategies, built to plug into Alpaca. This is the **research instrument** — the
thing you use to find and validate an edge *before* any execution infrastructure
gets built. There is deliberately no live-order code here.

## Read this first (the honesty part)

1. **Synthetic results do not prove an edge.** The offline synthetic data source
   bakes a variance-risk-premium edge into the data by construction (implied vol
   is set above realized vol). A profitable premium-selling backtest on synthetic
   data proves the **engine is wired correctly** — nothing more. To test for a
   *real* edge, run against real history via `AlpacaDataSource`.
2. **Fills are where backtests lie.** The `FillModel.slippage_frac` knob controls
   how much of the bid/ask half-spread you concede on every trade. Tune it
   pessimistically. If a strategy only works at `slippage_frac=0` (mid fills), it
   does not work.
3. **On real data the spread itself is a guess.** Alpaca serves *no historical
   option quotes* — the endpoints are `/options/bars`, `/options/trades`, and
   `quotes/latest`/`snapshots`, the last two live-only at every tier. So
   `AlpacaDataSource` observes a daily *trade* bar and models the bid/ask around
   it. Two assumptions stack: `bar close ≈ mid`, and `half-spread ≈ SpreadModel`.
   Never read a single real-data run; read `--band`, which sweeps both. If the
   sign of the edge flips inside that grid, you have a fill assumption, not an
   edge.
4. **The templates are hypotheses, not recommendations.** Put credit spreads and
   iron condors are here because they *encode cleanly*, not because they're the
   trade. You own the strategy selection and the risk. This is not financial
   advice, and none of it should touch live capital until it has survived both
   real-data backtesting and a paper-trading period.

## What the numbers mean

For premium selling, **win rate is the least useful number** — it's high by
design. Watch **expectancy per trade** and **max drawdown**, because the entire
risk lives in the tail. The report deliberately shows avg-win vs avg-loss so the
"win small, lose big" asymmetry is visible.

## Layout

```
bsm.py            Black-Scholes pricing/greeks + implied-vol solver (European approx)
trade.py          Leg / Position / ClosedTrade — one sign convention, centralized
data.py           OptionChain + OptionsDataSource; Synthetic (offline) + Alpaca (real)
spread.py         Bid/ask model for the Alpaca path + live-snapshot calibration
cache.py          Disk cache + rate limiter for as-of chain reconstruction
strategies.py     Strategy interface + PutCreditSpread + IronCondor templates
engine.py         FillModel + Backtest event loop (manage -> enter -> mark equity)
metrics.py        Win rate, expectancy, profit factor, max drawdown, P&L distribution
run_backtest.py   Entry point: synthetic runs, sweeps, real-data runs, calibration

events.py         Live event sources: news, corporate actions, IV/term structure,
                  realized vol, earnings-via-CSV; live chain from real snapshots
weights.py        Event weight table + scoring. Every weight carries a provenance
scanner.py        Polling job: score events -> ranked suggestions -> log. No orders

execution.py      Sizing, risk limits, MLEG orders, kill switch. Only module that trades
notify.py         Twilio SMS + ntfy push; token-scoped reply grammar
alerts.py         Approval daemon: scan -> size -> alert -> reply -> re-price -> submit
```

## Run it (offline, no keys)

```bash
python run_backtest.py            # backtest both templates on synthetic data
python run_backtest.py --sweep    # grid-search put-credit-spread parameters
```

No dependencies for synthetic mode — pure standard library.

### Where the templates currently stand

Both fail the harness's own honesty test. Swept across the fill assumption on
synthetic data — data with a 3-point variance risk premium *handed to them*:

```
strategy              slip  trades  totalP&L  exp/trade     PF
--------------------------------------------------------------
put_credit_spread      0.0      65       793       12.2   1.47
put_credit_spread      0.5      65       337        5.2   1.18
put_credit_spread      1.0      64      -305       -4.8   0.86
iron_condor            0.0      51      1294       25.4   3.61
iron_condor            0.5      51       717       14.1   2.11
iron_condor            1.0      51        95        1.9   1.11
```

$337 on $25,000 over three years is 1.3% total, against cash paying ~13% over
the same period — and it goes negative once you cross the spread. Per
`engine.py`: *if a strategy only works at slippage_frac=0, it doesn't work.*
These are the starting hypotheses to beat, not a validated edge.

### And on real SPY history

`--real -s SPY --band`, 2025-02-10 → 2026-08-07, total P&L on $25k:

```
put_credit_spread          slip=0.0        slip=0.5        slip=1.0
  tight (=SPY, measured)    391 ( 42)       596 ( 44)       526 ( 46)
  optimistic                391 ( 42)      1241 ( 54)     -4483 ( 52)
  realistic                 391 ( 42)     -4487 ( 52)     -4993 ( 51)
  pessimistic               391 ( 42)     -4998 ( 51)     -9270 ( 43)

iron_condor                slip=0.0        slip=0.5        slip=1.0
  tight (=SPY, measured)    662 ( 32)       419 ( 32)       779 ( 31)
  optimistic                662 ( 32)        78 ( 31)     -1842 ( 30)
  realistic                 662 ( 32)     -1859 ( 30)     -4060 ( 32)
  pessimistic               662 ( 32)     -4071 ( 32)    -22655 ( 90)
```

Three things to read out of that grid:

- **`slip=0.0` is identical down every spread row** (391 ×4, 662 ×4). It should
  be — at mid fills the modelled spread width cannot matter. A useful check that
  the spread model is wired where it belongs and nowhere else.
- **Only the measured row survives.** `tight` is the one anchored to
  `--calibrate` (SPY's traded band quotes ~0.5% of mid). `realistic` at 4% is
  ~8× SPY's actual spread, so its deep losses describe a less liquid underlying,
  not this one. Judge SPY on the `tight` row.
- **The grid is non-monotonic in slippage** — `optimistic` runs 391 → 1241 →
  −4483, and trade counts move (42 → 54 → 52). More slippage should not make
  more money. Different fills trip profit-targets and stops at different times,
  which shifts every subsequent entry. Path dependence, not edge.

Detail on the measured row (`slip=0.5`):

```
                    put_credit_spread    iron_condor
trades                            44             32
total P&L                       $596           $419
return on capital               2.4%           1.7%
expectancy / trade             $13.6          $13.1
  standard error              ±$24.5         ±$29.6
  t-statistic                   0.55           0.44
  95% CI                  -$34 .. $62    -$45 .. $71
win rate                       81.8%          59.4%
avg win / avg loss       $68 / -$231   $112 / -$132
max drawdown                   -5.3%          -3.4%
```

**t = 0.55 and t = 0.44.** The confidence interval on expectancy spans zero in
both cases: these results are statistically indistinguishable from no edge at
all. 44 trades is far too few, and the 81.8% win rate with an avg loss 3.4× the
avg win is precisely the "win small, lose big" shape where the tail governs and
small samples flatter you.

Even taking the point estimate at face value, 2.4% over 18 months lost to
T-bills paying roughly 6% over the same stretch — while carrying a −5.3%
drawdown and open tail risk.

This is the harness working. It was built to tell you whether an edge exists,
and for these two templates on SPY over this window, the answer is: not one you
can distinguish from zero.

### Friction sweep: why widening the spread does not rescue it

`--friction` attacks the cost side instead of hunting for edge, on the theory
that friction is reducible by construction while edge is not. Run on the same
SPY window with the measured spread (`python run_backtest.py --friction -s SPY`):

```
 width  expire  take | trades  totP&L   exp$  credit |  fric%  open  close | exp/risk     t
     5      --  0.50 |     44     596   13.6      98 |   7.3%  4.0%   3.3% |    3.37%  0.55
     5      --    -- |     29   -1533  -52.9     104 |   7.2%  4.0%   3.2% |  -13.11% -1.38
     5    0.10  0.50 |     41     470   11.5     105 |   7.0%  3.8%   3.2% |    2.85%  0.40
    10      --  0.50 |     39     230    5.9     192 |   3.5%  2.0%   1.5% |    0.71%  0.12
    20      --  0.50 |     33    1432   43.4     331 |   1.8%  1.0%   0.8% |    2.72%  0.62
    30      --  0.50 |     33    1783   54.0     437 |   1.3%  0.7%   0.5% |    2.10%  0.68
    30    0.10  0.50 |     33    1800   54.5     436 |   1.3%  0.7%   0.5% |    2.12%  0.68
```

**The friction thesis is arithmetically correct and strategically useless.**
Widening 5 → 30 cuts friction from 7.3% of credit to 1.3%, a 5.6× reduction,
exactly as predicted — credit scales with width while spread cost scales with
leg count, which is 2 either way. That column is arithmetic and reliable.

It does not improve risk-adjusted return, because widening degrades the
credit-to-risk ratio faster than it saves friction:

```
width  credit  avg risk  credit/risk  friction%  fric/risk  gross exp/risk
    5      98       404        24.3%       7.3%      1.77%          5.14%
   30     437      2571        17.0%       1.3%      0.22%          2.32%
```

Going wide saves 1.55% of risk in friction and gives up 2.82% of risk in gross
edge. The further-OTM wing you buy is cheap, so extra width adds max loss much
faster than it adds credit. Net effect is negative. Raw P&L looks better at
30-wide ($1,783 vs $596) purely because you are risking 6× as much per trade —
which is why the sweep reports expectancy per dollar at risk.

Two secondary results:

- **The expire-worthless exit is a dead end for this structure.** It barely
  fires. With a 50% profit target most positions close before min_dte, and the
  ones that survive to 21 DTE are precisely the ones whose short delta is still
  above the abandon threshold — the exit reasons show `abandon:` where
  `min_dte<=` used to be, with only 3–4 trades reaching actual expiry.
- **The 50% profit target is doing real work.** Removing it is strongly negative
  at every width (−$1,533 at 5-wide, and it flips 20/30-wide from +$1,432/+$1,783
  down to +$798/+$861). Taking profits early is what keeps the position out of
  the tail.

Caveat that governs all of it: every t-statistic here is between 0.01 and 0.68.
The *differences between cells* are not significant either, so the apparent
ranking of exp/risk is itself mostly noise. Only the friction columns are
arithmetic rather than inference — read those, and treat the rest as
hypothesis generation.

## Point it at real Alpaca history

`AlpacaDataSource` is implemented (against alpaca-py 0.43.x). It places **no
orders** and needs only market-data access, so use a paper key pair.

```bash
pip install -r requirements.txt
export APCA_API_KEY_ID=...
export APCA_API_SECRET_KEY=...

python run_backtest.py --calibrate -s SPY    # fit the spread model to a live chain
python run_backtest.py --real -s SPY --band  # backtest across fill assumptions
```

Run `--calibrate` during market hours — outside RTH most contracts have no live
market to measure. Take the fitted `SpreadModel` numbers, widen them (today's
liquidity flatters 2024), and put them in `spread.py`.

The first `--real` run downloads a few hundred MB into `.cache/alpaca/` and
takes a while; every run after it is local. Delete the directory to refetch.

**How it reconstructs the past.** It bulk-loads the window once — underlying
bars, the full contract universe, every daily option bar — then serves each
as-of chain from memory. Per-day fetching would need thousands of calls against
a 200/min limit; this needs tens. Four constraints it handles that will
otherwise silently corrupt your results:

- **History floor: February 2024.** Earlier start dates raise rather than
  quietly return an empty window.
- **Survivorship.** `get_option_contracts` reports a contract's status *today*,
  so everything already expired reads as `inactive`. Querying only `active`
  deletes the entire past. Both statuses are queried and merged.
- **Unadjusted spot.** Underlying bars are fetched `RAW`, because strikes are in
  unadjusted terms and a split-adjusted underlying misaligns every moneyness
  calculation against the strikes it's compared to.
- **Liquidity.** A contract with no bar on a date did not trade that date, and
  is dropped rather than filled at theoretical value. This is the single biggest
  thing keeping real-data results honest, and it's why trade counts fall well
  below the synthetic run's.

`src.skipped` reports why contracts were dropped. A large
`deep_itm_no_timevalue` count is normal — those are deep-ITM contracts whose
penny-rounded price leaves no recoverable implied vol, and nobody trades them.

**The free tier is the *indicative* feed** — a 15-minute-delayed derivative of
OPRA, not the consolidated BBO. Real OPRA is a paid subscription. Know which one
produced your numbers.

## Add your own strategy

Subclass `Strategy` and implement two methods:

```python
def propose_entry(self, chain, open_count) -> Position | None
def manage(self, position, chain) -> str | None   # return a close reason, or None
```

The engine handles fills, mark-to-market, and expiry. You only decide what to
open and when to close. Every tunable (delta, width, DTE, profit-take, stop,
min-DTE) is a constructor arg so it can be swept.

## The intended sequence

```
research harness (this)  ->  real-data backtest  ->  paper trade  ->  THEN build
the staging + one-click-approval execution layer on the front of a validated
strategy.
```

The execution layer is the *last* thing you build, and it guards something real
only once the steps before it have earned it.

## Live event scanner (`scanner.py`)

A polling job that watches real-time events, scores them against a weight table,
and prints ranked trade candidates. **It places no orders and imports no
order-submission API** — `TradingClient` is used only for the market clock.

```bash
python scanner.py --weights              # the weight table and its provenance
python scanner.py --once -s SPY,QQQ,IWM  # one pass
python scanner.py --watch --interval 300 # poll every 5 min during RTH
python scanner.py --review               # score past suggestions vs outcomes
```

### What it can actually see

Verified against alpaca-py 0.43.x:

| event | source | available |
|---|---|---|
| news bursts | `/v1beta1/news` | yes |
| dividends, splits, mergers | `/v1beta1/corporate-actions` | yes |
| IV, greeks, real bid/ask | `/v1beta1/options/snapshots` | yes |
| realized vol, trend | `/v2/stocks/bars` | yes |
| **earnings dates** | — | **no endpoint, any tier** |
| **FOMC / CPI / NFP** | — | **no endpoint, any tier** |

Earnings is the dominant single-name options event and Alpaca has no feed for
it, so `EarningsCalendar` reads an `earnings.csv` you maintain:

```csv
symbol,earnings_date
AAPL,2026-10-29
```

A symbol missing from that file reports UNKNOWN, and unknown is **vetoed rather
than assumed to be "no earnings"** — being short premium through an unexpected
print is exactly the tail these defined-risk structures exist to bound. Index
and broad-ETF symbols are exempt (`NO_EARNINGS_SYMBOLS`).

Unlike the backtest path, the scanner reads **real bid/ask and real greeks** from
live snapshots. The spread modelling that handicaps `--real` does not apply here.

### The weights are mostly guesses, and the code says so

Every factor carries a `Provenance`: `MEASURED` (backtested here), `LITERATURE`
(documented effect, direction solid, magnitude not), or `PRIOR` (reasoned
guess). Run `--weights` for the breakdown. Currently **88% of total weight is
not measured against real outcomes.**

Three design rules keep that from becoming astrology:

1. **Uncomputable factors are dropped, never defaulted.** Weights renormalize
   and the score reports its own `coverage`. A missing factor must not become a
   silent mild endorsement. `iv_rank` needs 60 daily observations, so on a fresh
   install it is simply absent and coverage sits at 78%.
2. **Vetoes are separate from weights.** Some conditions are disqualifying, not
   outvotable. This was learned the hard way: on the first live scan IWM topped
   the board at +0.313 while quoting a round-trip spread of 24% of the credit,
   because a strong VRP reading outvoted the −0.07 that fill quality
   contributed. These templates' expectancy is roughly 5% of credit, so that
   trade could not pay for itself. Fill cost above 15% of credit is now a veto.
3. **Every suggestion is logged with its arithmetic** to `suggestions.jsonl`,
   with an empty outcome slot. `--review` fills those in from later price
   history and reports breach rate by score bucket. If breach rate does not fall
   as composite rises, the weights are not earning their keep. That log is the
   only path from PRIOR to MEASURED.

### Known limitation: single-snapshot noise

Fill quality is measured from one snapshot, and on less liquid names it moves.
Two scans a minute apart read IWM's spread cost at 24% then 9%, and SPY's at 4%
then 12% — so the veto can flap between polls. Treat any single poll's spread
reading as one draw, not the truth. The log keeps every observation.

### Running it as a background job on macOS

```bash
# foreground, simplest
python scanner.py --watch --interval 300
```

For a launchd job, point a plist at `scanner.py --watch`, set
`APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` in `EnvironmentVariables`, and set
`WorkingDirectory` to this repo so `suggestions.jsonl` and `.cache/` land here.

## Alerting + staged execution (`alerts.py`)

The full loop: scan → score → size → SMS → your reply → re-price → submit.

```bash
python alerts.py --once --dry-run          # console output, no SMS, no orders
python alerts.py --sms                     # real texts, still cannot submit
python alerts.py --sms --arm               # confirmed orders -> PAPER
python alerts.py --sms --arm --live-money  # confirmed orders -> LIVE
```

`--arm` and `--live-money` are deliberately separate. Arming enables submission;
`--live-money` chooses whose money. Neither implies the other, neither is a
default, and the daemon checks the account number against the flag — pointing
paper keys at `--live-money` (or the reverse) exits rather than guessing.

### The text

```
[YCY] SPY put credit spread
756/751p exp Sep 25
SIZE 1 contract(s)
credit $131 | risk $369
short delta 0.30 | spread 4% of credit
score +0.25 (coverage 78%, mostly unvalidated)
sizing: budget $500 / $369 risk per contract
--
Reply Y YCY to confirm, N YCY to skip,
or 'YCY <n>' for n contracts.
Expires in 30 min.
```

Replies must carry the token. A bare `Y` does nothing, and neither does a token
you don't recognise. This inbox authorizes real orders, so an unscoped "yes"
must never be able to confirm whatever happens to be pending — a wrong number,
carrier spam, or a delayed duplicate of an older reply would all qualify.
An altered size is a *request*: it is re-validated against the caps, and
`YCY 5` against a 2-contract ceiling submits 2 and tells you so.

### Sizing

Off **defined risk**, not notional and not buying power — for a credit spread
the real exposure is `(width − credit)`, which is the number that shows up if
it goes wrong:

```
contracts = floor(min(equity × risk_frac, max_risk_per_trade) / max_loss_per_contract)
```

At the defaults ($25k, 2%, $500 budget) a 5-wide SPY spread risking ~$400/contract
sizes to 1. Then capped by `max_contracts`, remaining portfolio-risk headroom,
and the daily new-risk ceiling.

### Controls

Each is independent and any one alone stops an order:

| control | default |
|---|---|
| kill switch — `touch KILL_SWITCH` halts submission, no restart | off |
| armed flag — submission off unless `--arm` | disarmed |
| live opt-in — separate `--live-money` flag, account number verified | paper |
| max contracts / max risk per trade | 2 / $750 |
| max total open risk | $2,500 |
| max orders per day / new risk per day | 3 / $1,500 |
| approval TTL | 30 min |
| re-price tolerance | 10% |
| idempotency — `client_order_id` from the token | one order per token |
| limit orders only, never market | always |

**The re-price gate is the one that matters most.** A text quoting $131 of
credit, confirmed 40 minutes later, must not submit at the old price. The market
is re-read at submission and the order is abandoned if credit slipped past
tolerance. An approval is consent to a price, not a standing instruction.

`limit_slippage_frac` controls how hard you cross: `1.0` (default) prices at
`short.bid − long.ask` — marketable, near-certain fill, and you concede the full
spread the friction sweep measured at 7.3% of credit. `0.5` prices at the mid
and halves that, at the risk of not filling. Same knob as `FillModel.slippage_frac`
in the backtest, so what you sweep there is what you get here.

### Alpaca's paper fills are not real — do not validate on them

A probe order confirmed the MLEG construction works, and also that **the paper
engine filled a spread whose market credit was $1.04 at $3.12** — it fills at
your limit rather than at the market. Paper P&L from Alpaca will flatter any
premium-selling strategy enormously, and cannot be used to validate fills. The
scanner's logged *quotes* are real and still worth accumulating; paper *fills*
are fiction.

### Notification channels

Two are implemented. `--ntfy` works today with no account; `--sms` needs a
funded, verified Twilio account.

#### ntfy.sh (`--ntfy`) — free, no account, no carrier registration

```bash
python notify.py --new-topics          # generates two unguessable topic names
export NTFY_TOPIC_ALERTS="alpaca-alerts-..."
export NTFY_TOPIC_REPLIES="alpaca-reply-..."
```

Install the ntfy app and subscribe to the **alerts** topic only. Proposals
arrive as a push with tappable **Confirm** / **Skip** buttons; tapping one POSTs
the reply to the second topic, which the daemon polls. Nothing on your machine
is ever exposed.

**Security caveat, and it is a real one.** A public ntfy.sh topic has no sender
authentication — anyone who knows the reply topic name can confirm a trade. All
that protects you is an unguessable topic (why `--new-topics` uses 128 bits) and
a 3-character token that expires in 30 minutes. That is fine for alerts and fine
for paper. For `--arm --live-money` it is materially weaker than SMS from a
verified handset, and `alerts.py` warns when it sees that combination. Use
Twilio, a self-hosted ntfy with auth, or a paid tier with access control before
pointing real money at it.

#### Twilio SMS (`--sms`)

```bash
export TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=...
export TWILIO_FROM=+1XXXXXXXXXX      # your Twilio number
export ALERT_TO=+1XXXXXXXXXX         # your phone
```

Replies are collected by polling Twilio's inbound API, so nothing on your
machine needs to be publicly reachable. Only messages from `ALERT_TO` are
honoured — which is the security advantage over ntfy.

Three things a US Twilio account needs before any message arrives:

- **A funded account.** A trial with a negative balance is suspended, and every
  API call returns `401 code 20003` — which looks exactly like a bad token.
- **A number that can do A2P.** Toll-free numbers need toll-free verification;
  a rejected verification means messages are accepted by the API and silently
  dropped by carriers. A local 10DLC number needs A2P registration.
- **Verified recipients while on trial.** Trial accounts can only message
  pre-verified numbers.

### Running it periodically

`com.alpaca.alerts.plist` runs `--once` every 5 minutes rather than holding a
long-lived process, so a crash costs one cycle instead of the whole job. Fill in
the credentials, then:

```bash
cp com.alpaca.alerts.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.alpaca.alerts.plist
launchctl unload ~/Library/LaunchAgents/com.alpaca.alerts.plist   # to stop
```

`WorkingDirectory` must stay pointed at this repo — `pending.json`,
`orders.jsonl` and `KILL_SWITCH` all resolve relative to it. The shipped plist
omits `--arm` on purpose: installing the job should not by itself grant it the
ability to trade.

### Live requires different keys

Live trading uses a **separate** key pair from the live dashboard, and your live
account needs its own options approval — spreads are Level 2+, and paper showing
Level 3 does not carry over. The daemon refuses to run if the account type and
the flags disagree.

## AWS deployment (`aws/`)

Three container Lambdas on schedules, DynamoDB for state, SSM for secrets and
runtime flags. Deployed twice — `dev` against paper credentials, `prod` against
live ones.

```
aws/template.yaml         SAM: Lambdas, EventBridge crons, DynamoDB, alarms
aws/handlers.py           refresh / scan / respond entry points
aws/dynamo_store.py       DynamoDB + SSM implementations of storage.py
aws/seed_params.sh        one-time SSM parameter seeding, per environment
Dockerfile                container image (see "why a container" below)
storage.py                the seam: file-backed locally, DynamoDB in Lambda
```

| Lambda | Schedule | Job | Can trade? |
|---|---|---|---|
| `alpaca-refresh-<env>` | daily 07:30 ET | corporate actions → cache | no |
| `alpaca-scan-<env>` | every 5 min, RTH | quotes, chain, news, score, size → propose | no |
| `alpaca-respond-<env>` | every 1 min | poll replies, re-price, submit | **yes** |

`scan` builds its `Executor` with `armed=False` unconditionally, so it cannot
place an order even if its own logic were wrong. `respond` carries
`ReservedConcurrentExecutions: 1` so two submitters can never race.

### Deploy

```bash
export ALPACA_KEY=... ALPACA_SECRET=... NTFY_ALERTS=... NTFY_REPLIES=...
cd aws && ./seed_params.sh dev
sam build && sam deploy --config-env dev
```

The stack seeds `armed=off`, `live_money=off`, `kill=off`. It will alert you and
refuse every confirmation until you deliberately arm it.

### Why a container and not a zip

`alpaca-py` hard-depends on pandas and imports it at module load, putting the
dependency set at ~230MB against Lambda's 250MB unzipped limit. A container has
a 10GB limit, so this needs no changes to the data layer. Image is ~672MB; cold
start is a few seconds, which is irrelevant on a five-minute cron.

### The control surface is SSM, not the deployment

```bash
aws ssm put-parameter --name /alpaca/dev/kill  --value on --overwrite   # halt now
aws ssm put-parameter --name /alpaca/dev/armed --value on --overwrite   # allow orders
aws ssm put-parameter --name /alpaca/dev/live_money --value on --overwrite
```

Both flags are read per-invocation, so changes take effect within a minute and
**turning trading off never depends on a deploy succeeding**.

**The kill switch is deliberately not a CloudFormation resource.** A
CFN-managed parameter is reset to its template value on every deploy — you hit
the emergency stop during an incident, someone ships an unrelated change, and
the deploy silently sets it back to `off`. An emergency stop that a routine
deploy can undo is not an emergency stop. It is owned by `seed_params.sh`
instead, so it survives deploys, stack updates, and stack deletion.
`SsmKillSwitch` also fails **closed**: an unreadable parameter reports engaged,
so deleting it halts trading rather than enabling it.

### State layout (single DynamoDB table)

```
pk                    sk                  purpose
PENDING               <token>             proposal awaiting confirmation (TTL)
ORDER#<yyyy-mm-dd>    <iso-ts>#<token>    audit record, day-partitioned
TOKEN                 <token>             idempotency marker, conditional write
IV#<SYMBOL>           <yyyy-mm-dd>        daily ATM IV for the iv_rank factor
CACHE                 <key>               refresh-lambda output
```

Proposal expiry is a DynamoDB TTL rather than sweep code, and `PENDING` is also
filtered on read — TTL deletion lags by minutes, and an expired proposal must
never be actionable in the gap. Orders are day-partitioned so the daily-cap
query stays a single-partition read after a year of history.

## Phone console (`reply.html`)

A single self-contained HTML file: read-only dashboard plus the reply controls.
Not served from anywhere — keep it on your own device.

**Dashboard** (read-only, from the `stats` Lambda Function URL):
system posture pills, day/week/month/year P&L with an intraday sparkline,
pending proposals with inline Confirm/x1/Skip buttons, open positions with
unrealized P&L, risk-usage meters, and today's activity *including refusals* —
which never reach the broker and so are invisible in the Alpaca account, yet are
usually the answer to "why did nothing trade today".

**Customization** — gear icon, persisted to `localStorage`: endpoint, token,
reply topic, auto-refresh interval (off/15s/30s/60s/5m), and per-section
visibility. Nothing is baked in beyond the generated defaults, so the same file
works on a second device by re-entering the endpoint.

### Why a backend endpoint exists at all

Alpaca sends no CORS headers, and embedding API keys in a page you carry around
would hand trading credentials to anyone who picked up the file. So `stats.py`
runs as a Lambda behind a Function URL and the page holds only a read-token.

`stats.py` **cannot trade**: it never constructs an `Executor` and never imports
an order request type. The Function URL is public with a bearer token rather
than IAM, so it is written assuming the token eventually leaks — the worst case
is somebody reads your P&L, not that they move your money. Rotate with
`aws ssm put-parameter --name /alpaca/<env>/stats_token --value <new> --type SecureString --overwrite`.

### One piece of arithmetic worth not repeating

Alpaca's portfolio-history response has a `profit_loss` array and summing it
looks obvious. It is wrong — each entry is that point's P&L relative to
`base_value`, not an increment. On this account summing gave **-$900,009** for a
week whose true change was **-$9.24**. Period P&L is `equity[-1] - base_value`.

## Exit management (`handlers.manage`)

`scan` opens and `respond` submits — but for a while nothing **closed** anything.
That is not a missing convenience, it is a different strategy. The backtested
edge lives in the exits: 32 of 44 closes were profit-target hits, and the
friction sweep showed that removing the profit target turns **+$596 into
−$1,533**. A system that only opens is "hold every spread to expiry with no
stop", which is measurably worse than the thing that was tested.

`alpaca-manage-<env>` runs on the scan schedule and applies the same three rules
the backtest used, read from SSM so they can be tuned without a deploy:

```
profit_take = 0.50     close at 50% of credit captured
stop_mult   = 2.0      close at 2x credit lost
min_dte     = 21       roll off with 21 days left
```

**Exits are not confirmed by you.** Entries ask permission; exits do not. A
stop-loss that waits for a tap is not a stop-loss — the moment it matters most
is the moment you are in a meeting. Same asymmetry as HALT/RESUME: the
risk-reducing direction gets less friction. Every exit still notifies, is
journalled, and respects the kill switch and the day halt.

Entry credit is recorded at submit time (`POSITION#<token>` in DynamoDB) because
the broker reports legs but not what credit you collected, and both exit rules
are multiples of that credit.

`scan` also now skips a symbol that already has an open position. The backtest
ran `max_concurrent=1`; without this the live system would stack correlated
spreads on one underlying, which is a risk profile nothing measured.

## Tests

```bash
python test_suite.py        # 58 tests, no network, no AWS
```

Weighted toward the safety controls rather than the pricing math, because every
bug that actually shipped was in the plumbing. Tests named `test_bug_*` pin
defects that reached a running system so they cannot return quietly.

## Purchased historical quotes (`vendor_data.py`)

The single change that converts this from "cannot distinguish from zero" to an
answerable question.

`AlpacaDataSource` cannot observe a historical bid/ask — no as-of quote endpoint
exists at any tier — so it reads a daily trade bar and **models** the spread.
Every number it produces is conditional on that model, which is why results are
reported as a grid. Alpaca's history also begins February 2024, capping the
usable window at roughly 20 independent 45-day episodes against the ~570 needed.

A purchased file fixes both. The bid and ask are the ones that existed, and a
decade of history gives ~110 independent episodes instead of 20.

```bash
python vendor_data.py --inspect ~/Downloads/spy_options_2012_2026.csv   # ALWAYS FIRST
```

That prints the file's headers and scores each preset against them. A wrong
preset yields zero usable rows and a confusing failure later, so check the fit
before indexing.

```python
from vendor_data import VendorFileDataSource
src = VendorFileDataSource("~/Downloads/spy.csv", "SPY", preset="cboe").prepare()
```

Presets ship for **cboe**, **orats**, **ivolatility** and **canonical**. They are
plain dictionaries mapping canonical field names to vendor headers — adding a
vendor is a few lines, not a new class.

### Which vendor

[Cboe DataShop "Option EOD Summary"](https://datashop.cboe.com/option-eod-summary)
carries NBBO bid/ask per series back to January 2012 and sells as a **one-time
purchase** rather than a subscription, which suits a one-off research question.

Buy **SPY, not SPX**: index bid/ask requires a separate Cboe Global Indices
licence starting around $1k/month; ETF options do not. SPY is what this harness
trades anyway.

### What changes once you have it

- `spread_model` is unused — the spread is observed
- `--band` becomes unnecessary; there is no assumption left to sweep
- `slippage_frac` still applies, because how hard *you* cross a real spread is
  still a choice
- One-sided and crossed markets are dropped rather than filled. A quote you
  could not have traded must never become a fill.

Files are indexed once into per-date shards. A decade of option quotes is tens
of millions of rows and will not fit in memory; any single trading day will.
