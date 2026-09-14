"""Regression suite. Run: python test_suite.py

Weighted toward the SAFETY controls rather than the pricing math, because the
bugs that actually shipped were all in the plumbing: file modes, cached config,
replayed commands, a veto that could never fire. The Black-Scholes has been
right the whole time.

Every test named test_bug_* pins a defect that reached a running system. Those
exist so the same mistake cannot return quietly.

Stdlib unittest, no pytest, no network, no AWS. Anything needing DynamoDB is
skipped unless moto is installed.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import bsm
import execution as ex
import notify as nt
import storage
import weights as W
from data import OptionQuote, SyntheticDataSource
from engine import Backtest, BacktestConfig, FillModel
from strategies import IronCondor, PutCreditSpread
from trade import Leg, Position


def q(strike, bid, ask, right="put", delta=-0.30, sym=None, expiry=None):
    return OptionQuote(right, strike, expiry or date(2026, 9, 25),
                       bid, ask, delta, 0.15, symbol=sym)


# --------------------------------------------------------------------------- #
class TestPricing(unittest.TestCase):
    def test_implied_vol_roundtrip(self):
        for S, K, T, r, sig, right in [
                (100, 100, .12, .04, .20, "call"), (100, 90, .12, .04, .25, "put"),
                (450, 400, .05, .04, .35, "put"), (50, 60, .25, .04, .60, "call"),
                (773, 756, .12, .04, .17, "put")]:
            p = bsm.bs_price(S, K, T, r, sig, right)
            iv = bsm.implied_vol(p, S, K, T, r, right)
            self.assertIsNotNone(iv, f"no solution for {right} {S}/{K}")
            self.assertAlmostEqual(iv, sig, places=3)

    def test_implied_vol_rejects_impossible_prices(self):
        self.assertIsNone(bsm.implied_vol(0.5, 100, 110, .1, .04, "put"))   # < intrinsic
        self.assertIsNone(bsm.implied_vol(200, 100, 110, .1, .04, "put"))   # > bracket
        self.assertIsNone(bsm.implied_vol(0, 100, 110, .1, .04, "put"))
        self.assertIsNone(bsm.implied_vol(5, 100, 110, 0, .04, "put"))      # expired

    def test_put_delta_is_negative(self):
        d = bsm.bs_delta(100, 95, .12, .04, .2, "put")
        self.assertLess(d, 0)
        self.assertGreater(d, -1)


class TestTradeMath(unittest.TestCase):
    def setUp(self):
        self.pos = Position([
            Leg("put", 294, date(2026, 9, 25), "sell", 1, 5.00),
            Leg("put", 289, date(2026, 9, 25), "buy", 1, 4.09)], "IWM", date(2026, 8, 12))

    def test_credit_sign(self):
        self.assertAlmostEqual(self.pos.credit_received, 91.0, places=6)

    def test_max_loss_is_width_minus_credit(self):
        self.assertAlmostEqual(self.pos.max_loss, 500 - 91.0, places=6)

    def test_pnl_zero_when_closed_at_entry(self):
        prices = {(l.right, l.strike): l.entry_price for l in self.pos.legs}
        self.assertAlmostEqual(
            self.pos.open_pnl(lambda l: prices[(l.right, l.strike)]), 0.0, places=6)


class TestFillsAndFriction(unittest.TestCase):
    def test_slippage_direction(self):
        f = FillModel(slippage_frac=1.0)
        quote = q(294, 5.00, 5.20)
        self.assertAlmostEqual(f.fill_price(quote, "sell"), 5.00)   # sell at bid
        self.assertAlmostEqual(f.fill_price(quote, "buy"), 5.20)    # buy at ask

    def test_friction_is_tracked(self):
        data = SyntheticDataSource(start=date(2024, 1, 1), end=date(2024, 6, 30), seed=3)
        cfg = BacktestConfig(underlying="SYN", start=date(2024, 1, 1),
                             end=date(2024, 6, 30), starting_cash=25_000)
        bt = Backtest(data, PutCreditSpread(), cfg, FillModel(slippage_frac=0.5)).run()
        self.assertGreater(bt.credit_collected, 0)
        self.assertGreater(bt.spread_cost, 0)
        self.assertAlmostEqual(bt.spread_cost,
                               bt.spread_cost_open + bt.spread_cost_close, places=6)
        self.assertTrue(0 < bt.friction_frac < 1)

    def test_zero_slippage_costs_nothing(self):
        data = SyntheticDataSource(start=date(2024, 1, 1), end=date(2024, 6, 30), seed=3)
        cfg = BacktestConfig(underlying="SYN", start=date(2024, 1, 1),
                             end=date(2024, 6, 30), starting_cash=25_000)
        bt = Backtest(data, PutCreditSpread(), cfg, FillModel(slippage_frac=0.0)).run()
        self.assertAlmostEqual(bt.spread_cost, 0.0, places=6)


class TestStrategyRules(unittest.TestCase):
    def test_bug_exdiv_veto_applies_only_to_short_calls(self):
        """Shipped bug: the ex-div veto was applied to put credit spreads, which
        carry no dividend assignment risk. It would have blocked nearly every
        trade for a risk that does not exist."""
        pcs = PutCreditSpread().propose_entry(_chain_with_puts(), 0)
        self.assertIsNotNone(pcs)
        self.assertFalse(any(l.action == "sell" and l.right == "call" for l in pcs.legs),
                         "a put credit spread must contain no short call")

    def test_expire_rule_holds_only_when_far_otm(self):
        s = PutCreditSpread(min_dte=21, expire_below_delta=0.10, abandon_delta=0.15)
        chain = _chain_at_delta(-0.04)                  # far OTM -> ride it out
        pos = Position([Leg("put", 280, chain.expiries()[0], "sell", 1, 1.0),
                        Leg("put", 275, chain.expiries()[0], "buy", 1, 0.5)],
                       "IWM", date(2026, 8, 12))
        chain.as_of = chain.expiries()[0] - timedelta(days=10)   # inside min_dte
        self.assertIsNone(s.manage(pos, chain), "far-OTM position should ride to expiry")

    def test_expire_rule_abandons_when_breached(self):
        s = PutCreditSpread(min_dte=21, expire_below_delta=0.10, abandon_delta=0.15)
        chain = _chain_at_delta(-0.40)                  # came back at us
        pos = Position([Leg("put", 280, chain.expiries()[0], "sell", 1, 1.0),
                        Leg("put", 275, chain.expiries()[0], "buy", 1, 0.5)],
                       "IWM", date(2026, 8, 12))
        chain.as_of = chain.expiries()[0] - timedelta(days=10)
        self.assertIn("abandon", s.manage(pos, chain) or "")


def _chain_at_delta(d):
    """Every leg at one delta -- for exercising manage(), which only reads the
    short leg's delta and never selects strikes."""
    from data import OptionChain
    exp = date(2026, 9, 25)
    quotes = [q(k, 1.0, 1.1, "put", d, f"IWM{k}", exp) for k in (275, 280)]
    return OptionChain(date(2026, 8, 12), "IWM", 300.0, quotes)


def _chain_with_puts(short_delta=-0.30):
    """Deltas must VARY with strike or select_by_delta picks the lowest strike
    and leg_at_offset then finds no wing beneath it."""
    from data import OptionChain
    exp = date(2026, 9, 25)
    ladder = {270: -0.05, 275: -0.10, 280: -0.18, 285: -0.24,
              289: -0.30, 294: -0.42, 300: -0.55}
    quotes = [q(k, 1.0, 1.1, "put", d, f"IWM{k}", exp) for k, d in ladder.items()]
    ch = OptionChain(date(2026, 8, 12), "IWM", 300.0, quotes)
    ch._short_delta_override = short_delta
    return ch


# --------------------------------------------------------------------------- #
class TestRiskControls(unittest.TestCase):
    """Every gate that stands between a bad input and a live order."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmp)
        storage.use(journal=storage.FileJournal(), kill=storage.FileKillSwitch(),
                    pending=storage.FilePendingStore(), iv_history=storage.FileIVHistory())
        self.L = ex.RiskLimits()
        self.trade = ex.size_trade("A3F", "IWM", None,
                                   q(294, 5.00, 5.10, sym="S"),
                                   q(289, 4.09, 4.19, sym="L"), self.L)

    def tearDown(self):
        os.chdir(self.cwd)

    def _executor(self, armed=True, is_open=True, s_bid=5.00, l_ask=4.19):
        class T:
            def __init__(s): s.n = 0
            def get_all_positions(s): return []
            def get_clock(s): return SimpleNamespace(is_open=is_open)
            def submit_order(s, r):
                s.n += 1
                return SimpleNamespace(id=f"ord-{s.n}", status="accepted")
        class O:
            def get_option_latest_quote(s, r):
                return {"S": SimpleNamespace(bid_price=s_bid, ask_price=s_bid + 0.10),
                        "L": SimpleNamespace(bid_price=l_ask - 0.10, ask_price=l_ask)}
        t = T()
        return ex.Executor(t, O(), self.L, armed=armed), t

    # ---- sizing refusals ------------------------------------------------ #
    def test_refuses_quotes_without_occ_symbol(self):
        """Structural guarantee: no order can be built from backtest data."""
        with self.assertRaises(ex.RiskRefusal):
            ex.size_trade("X", "IWM", None, q(294, 5.0, 5.1), q(289, 4.0, 4.1), self.L)

    def test_refuses_no_credit(self):
        with self.assertRaises(ex.RiskRefusal):
            ex.size_trade("X", "IWM", None, q(294, 4.0, 4.1, sym="S"),
                          q(289, 4.3, 4.4, sym="L"), self.L)

    def test_refuses_credit_below_floor(self):
        with self.assertRaises(ex.RiskRefusal):
            ex.size_trade("X", "IWM", None, q(294, 4.20, 4.25, sym="S"),
                          q(289, 4.05, 4.10, sym="L"), self.L)

    def test_refuses_spread_wider_than_cap(self):
        with self.assertRaises(ex.RiskRefusal):
            ex.size_trade("X", "IWM", None, q(294, 5.00, 5.60, sym="S"),
                          q(289, 4.00, 4.60, sym="L"), self.L)

    def test_portfolio_cap_blocks_when_full(self):
        with self.assertRaises(ex.RiskRefusal):
            ex.size_trade("X", "IWM", None, q(294, 5.0, 5.1, sym="S"),
                          q(289, 4.09, 4.19, sym="L"), self.L,
                          open_risk=self.L.max_open_risk)

    def test_sizes_off_defined_risk(self):
        self.assertEqual(self.trade.contracts, 1)
        # credit = short.bid 5.00 - long.ask 4.19 = $81, so max loss = 500 - 81
        self.assertAlmostEqual(self.trade.credit_per_contract, 81.0, places=2)
        self.assertAlmostEqual(self.trade.max_loss_per_contract, 419.0, places=2)

    # ---- submission gates ------------------------------------------------ #
    def test_disarmed_refuses(self):
        e, t = self._executor(armed=False)
        with self.assertRaises(ex.RiskRefusal):
            e.submit(self.trade)
        self.assertEqual(t.n, 0)

    def test_kill_switch_refuses(self):
        storage.KILL.path.touch()
        e, t = self._executor()
        with self.assertRaises(ex.RiskRefusal):
            e.submit(self.trade)
        self.assertEqual(t.n, 0)

    def test_market_closed_refuses(self):
        e, t = self._executor(is_open=False)
        with self.assertRaises(ex.RiskRefusal):
            e.submit(self.trade)
        self.assertEqual(t.n, 0)

    def test_stale_approval_refuses(self):
        old = ex.SizedTrade(**{**self.trade.__dict__,
                               "created_at": (datetime.now(timezone.utc)
                                              - timedelta(hours=2)).isoformat()})
        e, t = self._executor()
        with self.assertRaises(ex.RiskRefusal):
            e.submit(old)
        self.assertEqual(t.n, 0)

    def test_reprice_gate_refuses_when_credit_moved(self):
        e, t = self._executor(s_bid=4.20)     # credit collapsed since the alert
        with self.assertRaises(ex.RiskRefusal):
            e.submit(self.trade)
        self.assertEqual(t.n, 0)

    def test_happy_path_submits_a_limit_order(self):
        e, t = self._executor()
        out = e.submit(self.trade)
        self.assertEqual(out["status"], "submitted")
        self.assertEqual(t.n, 1)
        self.assertGreater(out["limit_price"], 0)

    def test_idempotent_on_token(self):
        e, t = self._executor()
        e.submit(self.trade)
        with self.assertRaises(ex.RiskRefusal):
            e.submit(self.trade)
        self.assertEqual(t.n, 1, "one token must never produce two orders")

    def test_daily_order_cap_when_configured(self):
        """The count cap is optional now, but must still bite when set."""
        self.L.max_orders_per_day = 3
        self.L.max_new_risk_per_day = 10_000     # isolate the count cap
        e, t = self._executor()
        for i in range(3):
            e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": f"D{i}"}))
        with self.assertRaises(ex.RiskRefusal) as cm:
            e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "OVER"}))
        self.assertIn("order cap", str(cm.exception))
        self.assertEqual(t.n, 3)

    def test_order_count_is_uncapped_by_default(self):
        """Trade COUNT is no longer a control. Capital is.

        The count cap measured the wrong thing: closes consumed it too, so a
        day of exits locked out entries while barely any money was at risk.
        """
        self.assertIsNone(ex.RiskLimits().max_orders_per_day)
        self.L.max_new_risk_per_day = 10_000
        e, t = self._executor()
        for i in range(9):
            e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": f"N{i}"}))
        self.assertEqual(t.n, 9, "nine orders must not trip a count cap")

    # ---- daily capital budget ------------------------------------------ #
    def _book_close(self, token, pnl):
        """Journal a closed position with a realised result."""
        storage.JOURNAL.append({
            "token": f"CLOSE-{token}", "status": "submitted", "total_risk": 0.0,
            "realized_pnl": pnl,
            "submitted_at": datetime.now(timezone.utc).isoformat()})

    def test_daily_capital_cap_blocks_the_trade_that_would_exceed_it(self):
        self.L.max_new_risk_per_day = 1000.0
        e, t = self._executor()
        r = self.trade.total_risk
        self.assertLess(2 * r, 1000.0, "fixture must allow two trades")
        self.assertGreater(3 * r, 1000.0, "fixture must block the third")
        e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "C1"}))
        e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "C2"}))
        with self.assertRaises(ex.RiskRefusal) as cm:
            e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "C3"}))
        self.assertIn("daily capital cap", str(cm.exception))
        self.assertEqual(t.n, 2)

    def test_realised_profit_frees_daily_budget(self):
        """A closed winner hands the capital back, so the day may redeploy it."""
        self.L.max_new_risk_per_day = 1000.0
        e, t = self._executor()
        e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "P1"}))
        e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "P2"}))
        with self.assertRaises(ex.RiskRefusal):
            e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "P3"}))
        self._book_close("P1", 500.0)            # a winner lands
        out = e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "P4"}))
        self.assertEqual(out["status"], "submitted")
        self.assertEqual(t.n, 3)

    def test_realised_loss_tightens_daily_budget(self):
        """Symmetric by design: a losing day gets less rope, not the same rope."""
        self.L.max_new_risk_per_day = 1000.0
        e, t = self._executor()
        e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "L1"}))
        self._book_close("L1", -300.0)
        # One trade deployed plus a $300 loss consumes enough that a second
        # trade, which would otherwise fit, no longer does.
        self.assertLess(2 * self.trade.total_risk, 1000.0)
        with self.assertRaises(ex.RiskRefusal) as cm:
            e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "L2"}))
        self.assertIn("daily capital cap", str(cm.exception))
        self.assertEqual(t.n, 1)

    def test_closes_do_not_consume_deployment(self):
        """A close carries total_risk 0: exiting must never eat entry budget."""
        self.L.max_new_risk_per_day = 1000.0
        e, t = self._executor()
        for i in range(6):
            self._book_close(f"X{i}", 0.0)
        out = e.submit(ex.SizedTrade(**{**self.trade.__dict__, "token": "AFTER"}))
        self.assertEqual(out["status"], "submitted")

    def test_dry_run_needs_no_arming(self):
        """Shipped bug: the armed check ran before the dry_run branch, so
        --dry-run could never show what it would do."""
        e, t = self._executor(armed=False)
        out = e.submit(self.trade, dry_run=True)
        self.assertEqual(out["status"], "dry_run")
        self.assertEqual(t.n, 0)

    def test_refusal_does_not_block_later_retry(self):
        e, _ = self._executor(armed=False)
        with self.assertRaises(ex.RiskRefusal):
            e.submit(self.trade)
        e2, t2 = self._executor(armed=True)
        out = e2.submit(self.trade)
        self.assertEqual(out["status"], "submitted")
        self.assertEqual(t2.n, 1)


# --------------------------------------------------------------------------- #
class TestReplyGrammar(unittest.TestCase):
    def test_bug_bare_yes_must_not_confirm(self):
        """This inbox authorizes real orders. An unscoped 'Y' must do nothing."""
        self.assertIsNone(nt.parse_reply("Y", ["A3F"]))
        self.assertIsNone(nt.parse_reply("yes please", ["A3F"]))
        self.assertIsNone(nt.parse_reply("ok", ["A3F"]))

    def test_unknown_token_ignored(self):
        self.assertIsNone(nt.parse_reply("Y ZZZ", ["A3F"]))

    def test_confirm_and_decline(self):
        self.assertTrue(nt.parse_reply("Y A3F", ["A3F"]).confirmed)
        self.assertFalse(nt.parse_reply("N A3F", ["A3F"]).confirmed)

    def test_quantity_forms(self):
        for body in ("A3F 2", "2 A3F", "a3f 2"):
            r = nt.parse_reply(body, ["A3F"])
            self.assertTrue(r.confirmed)
            self.assertEqual(r.contracts, 2)

    def test_zero_quantity_is_a_decline(self):
        self.assertFalse(nt.parse_reply("A3F 0", ["A3F"]).confirmed)

    def test_carrier_spam_ignored(self):
        self.assertIsNone(nt.parse_reply("Sale! Text STOP to opt out", ["A3F"]))

    def test_token_alphabet_excludes_ambiguous_chars(self):
        toks = {nt.new_token() for _ in range(3000)}
        self.assertFalse([t for t in toks if set(t) & set("01OIL")])

    def test_halt_needs_no_token_resume_does(self):
        self.assertEqual(nt.parse_command("HALT").kind, "halt")
        self.assertIsNone(nt.parse_command("RESUME"))
        self.assertIsNone(nt.parse_command("RESUME ZZZ", "PY6"))
        self.assertEqual(nt.parse_command("RESUME PY6", "PY6").kind, "resume")

    def test_trade_reply_is_not_a_command(self):
        self.assertIsNone(nt.parse_command("Y A3F", "PY6"))


# --------------------------------------------------------------------------- #
class TestScoring(unittest.TestCase):
    def test_missing_factors_are_dropped_not_defaulted(self):
        s = W.score({"iv_minus_rv": (0.5, ""), "iv_rank": (None, "no data")}, [])
        keys = {f.key: f.value for f in s.factors}
        self.assertIsNone(keys["iv_rank"])
        self.assertLess(s.coverage, 1.0)

    def test_coverage_reflects_computable_weight(self):
        full = W.score({k: (0.0, "") for k in W.DEFAULT_FACTORS}, [])
        self.assertAlmostEqual(full.coverage, 1.0, places=6)

    def test_no_factors_scores_zero_not_nan(self):
        s = W.score({}, [])
        self.assertEqual(s.composite, 0.0)
        self.assertEqual(s.coverage, 0.0)

    def test_bug_fill_cost_is_a_veto_not_a_weight(self):
        """Shipped bug: IWM ranked top of the board at +0.313 while quoting a
        round-trip spread of 24% of credit, because favourable factors outvoted
        fill quality. Expectancy is ~5% of credit; that trade could not pay for
        itself."""
        self.assertIsNotNone(W.spread_veto(24.0, 100.0))
        self.assertIsNone(W.spread_veto(4.0, 100.0))

    def test_vetoed_score_reports_vetoed(self):
        s = W.score({"iv_minus_rv": (1.0, "")}, ["earnings UNKNOWN"])
        self.assertTrue(s.vetoed)


# --------------------------------------------------------------------------- #
class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.cwd = os.getcwd(); os.chdir(self.tmp)
        self.j = storage.FileJournal()

    def tearDown(self):
        os.chdir(self.cwd)

    def test_journal_roundtrip(self):
        now = datetime.now(timezone.utc).isoformat()
        self.j.append({"token": "A3F", "status": "submitted",
                       "order_id": "o1", "submitted_at": now})
        self.assertEqual(len(self.j.todays_submitted()), 1)
        self.assertEqual(self.j.find_submitted("A3F")["order_id"], "o1")

    def test_refusals_do_not_count_as_submitted(self):
        now = datetime.now(timezone.utc).isoformat()
        self.j.append({"token": "B2C", "status": "refused", "submitted_at": now})
        self.assertEqual(self.j.todays_submitted(), [])
        self.assertIsNone(self.j.find_submitted("B2C"))

    def test_truncated_line_does_not_break_caps(self):
        self.j.path.write_text('{"token":"A","status":"submitted"}\n{"broken')
        self.j.todays_submitted()      # must not raise

    def test_pending_roundtrip(self):
        p = storage.FilePendingStore()
        p.put("A3F", {"underlying": "IWM"})
        self.assertIn("A3F", p.all())
        p.delete("A3F")
        self.assertNotIn("A3F", p.all())


# --------------------------------------------------------------------------- #
class TestOccAndDividends(unittest.TestCase):
    def test_occ_roundtrip(self):
        from spread import _parse_occ
        sym = ex.occ_symbol("IWM", date(2026, 9, 25), "put", 294)
        self.assertEqual(sym, "IWM260925P00294000")
        expiry, right, strike = _parse_occ(sym)
        self.assertEqual((expiry, right, strike), (date(2026, 9, 25), "put", 294.0))

    def test_occ_handles_fractional_strikes(self):
        from spread import _parse_occ
        sym = ex.occ_symbol("SPY", date(2026, 9, 25), "call", 773.5)
        self.assertEqual(_parse_occ(sym)[2], 773.5)

    def test_parse_occ_rejects_garbage(self):
        from spread import _parse_occ
        self.assertEqual(_parse_occ("not-a-symbol")[0], None)


# --------------------------------------------------------------------------- #
class TestSyntheticRegression(unittest.TestCase):
    """Pins the offline numbers. If these move, something in the engine changed."""

    def test_known_synthetic_results(self):
        data = SyntheticDataSource(start=date(2023, 1, 1), end=date(2025, 12, 31),
                                   s0=100.0, realized_vol=0.18, vrp=0.03,
                                   drift=0.06, seed=11)
        cfg = BacktestConfig(underlying="SYN", start=date(2023, 1, 1),
                             end=date(2025, 12, 31), starting_cash=25_000,
                             max_concurrent=1)
        fills = FillModel(slippage_frac=0.5)
        pcs = Backtest(data, PutCreditSpread(short_delta=0.30, wing_width=5,
                       target_dte=45, profit_take=0.50, stop_mult=2.0,
                       min_dte=21), cfg, fills).run()
        ic = Backtest(data, IronCondor(short_delta=0.20, wing_width=5,
                      target_dte=45, profit_take=0.50, stop_mult=2.0,
                      min_dte=21), cfg, fills).run()
        self.assertEqual(len(pcs.closed), 65)
        self.assertAlmostEqual(sum(t.pnl for t in pcs.closed), 337, delta=1)
        self.assertEqual(len(ic.closed), 51)
        self.assertAlmostEqual(sum(t.pnl for t in ic.closed), 717, delta=1)




# --------------------------------------------------------------------------- #
class TestExitRules(unittest.TestCase):
    """The half of the strategy that was missing until now.

    32 of 44 backtested exits were profit-target closes, and removing the
    profit target flipped +$596 to -$1,533. These rules ARE the edge.
    """

    def setUp(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "aws"))
        self.pos = {
            "token": "A3F", "underlying": "IWM",
            "expiry": (date.today() + timedelta(days=45)).isoformat(),
            "short_occ": "S", "long_occ": "L",
            "short_strike": 294.0, "long_strike": 289.0, "contracts": 1,
            "entry_credit_per_contract": 100.0, "max_loss_per_contract": 400.0,
        }
        self.cfg = {"profit_take": "0.50", "stop_mult": "2.0", "min_dte": "21"}

    def _clients(self, short_ask, long_bid):
        class O:
            def get_option_latest_quote(s, r):
                return {"S": SimpleNamespace(ask_price=short_ask, bid_price=short_ask-0.05),
                        "L": SimpleNamespace(bid_price=long_bid, ask_price=long_bid+0.05)}
        return {"option": O()}

    def _verdict(self, short_ask, long_bid, pos=None):
        import handlers
        return handlers._exit_verdict(pos or self.pos,
                                      self._clients(short_ask, long_bid),
                                      self.cfg, False)

    def test_holds_when_nothing_triggered(self):
        # cost to close 0.80 vs 1.00 credit -> +$20, below the 50% target
        self.assertIsNone(self._verdict(1.00, 0.20))

    def test_profit_target_fires(self):
        # cost to close 0.40 -> pnl +$60 >= 50% of $100
        v = self._verdict(0.50, 0.10)
        self.assertIsNotNone(v)
        self.assertIn("profit target", v[0])

    def test_stop_fires(self):
        # cost to close 3.10 -> pnl -$210 <= -2x credit
        v = self._verdict(3.50, 0.40)
        self.assertIsNotNone(v)
        self.assertIn("stop", v[0])

    def test_min_dte_fires_even_when_flat(self):
        near = dict(self.pos)
        near["expiry"] = (date.today() + timedelta(days=10)).isoformat()
        v = self._verdict(1.00, 0.20, near)
        self.assertIsNotNone(v)
        self.assertIn("min_dte", v[0])

    def test_no_quote_means_no_action(self):
        import handlers
        class O:
            def get_option_latest_quote(s, r): return {}
        self.assertIsNone(handlers._exit_verdict(self.pos, {"option": O()},
                                                 self.cfg, False))

    def test_profit_target_takes_precedence_over_min_dte(self):
        near = dict(self.pos)
        near["expiry"] = (date.today() + timedelta(days=5)).isoformat()
        v = self._verdict(0.50, 0.10, near)
        self.assertIn("profit target", v[0],
                      "a winner inside min_dte should book the profit, not roll off")


class TestPositionStacking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.cwd = os.getcwd(); os.chdir(self.tmp)
        storage.use(positions=storage.FilePositionStore())

    def tearDown(self):
        os.chdir(self.cwd)

    def test_position_store_roundtrip(self):
        storage.POSITIONS.put("A3F", {"underlying": "IWM"})
        self.assertEqual(list(storage.POSITIONS.all()), ["A3F"])
        storage.POSITIONS.delete("A3F")
        self.assertEqual(storage.POSITIONS.all(), {})

    def test_bug_submit_records_the_position(self):
        """Without this record nothing can manage the exit: the broker knows the
        legs but not the credit, and both exit rules are multiples of credit."""
        L = ex.RiskLimits()
        trade = ex.size_trade("A3F", "IWM", None, q(294, 5.00, 5.10, sym="S"),
                              q(289, 4.09, 4.19, sym="L"), L)
        class T:
            def get_all_positions(s): return []
            def get_clock(s): return SimpleNamespace(is_open=True)
            def submit_order(s, r): return SimpleNamespace(id="o1", status="accepted")
        class O:
            def get_option_latest_quote(s, r):
                return {"S": SimpleNamespace(bid_price=5.00, ask_price=5.10),
                        "L": SimpleNamespace(bid_price=4.09, ask_price=4.19)}
        ex.Executor(T(), O(), L, armed=True).submit(trade)
        rec = storage.POSITIONS.all().get("A3F")
        self.assertIsNotNone(rec, "submit must record the position for management")
        self.assertGreater(rec["entry_credit_per_contract"], 0)
        self.assertEqual(rec["short_occ"], "S")


# --------------------------------------------------------------------------- #
class TestVendorIngest(unittest.TestCase):
    """The path that replaces the modelled spread with a purchased one."""

    def setUp(self):
        import csv
        from vendor_data import VendorFileDataSource
        self.tmp = tempfile.mkdtemp()
        self.csv = os.path.join(self.tmp, "v.csv")
        with open(self.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["underlying_symbol","quote_date","expiration","strike",
                        "option_type","bid_eod","ask_eod","underlying_bid_eod",
                        "trade_volume","open_interest","delta","implied_volatility"])
            rows = [
                # tradeable
                ("SPY","2024-03-01","2024-04-19","470","P","5.00","5.10","480","100","500","-0.30","0.16"),
                ("SPY","2024-03-01","2024-04-19","465","P","4.00","4.10","480","100","500","-0.24","0.16"),
                # crossed market -- must be dropped
                ("SPY","2024-03-01","2024-04-19","460","P","4.20","4.10","480","50","100","-0.20","0.16"),
                # one-sided -- must be dropped
                ("SPY","2024-03-01","2024-04-19","455","P","0","0.90","480","50","100","-0.15","0.16"),
                # a different underlying -- must be ignored
                ("QQQ","2024-03-01","2024-04-19","400","P","3.00","3.10","410","10","20","-0.30","0.20"),
            ]
            for r in rows: w.writerow(r)
        self.src = VendorFileDataSource(self.csv, "SPY", preset="cboe",
                                        index_dir=os.path.join(self.tmp,"idx"),
                                        verbose=False).prepare()

    def test_indexes_only_the_requested_underlying(self):
        ch = self.src.get_chain("SPY", date(2024,3,1))
        self.assertIsNotNone(ch)
        self.assertEqual(ch.spot, 480.0)

    def test_drops_crossed_and_one_sided_markets(self):
        """A quote you could not have traded must not become a fill."""
        ch = self.src.get_chain("SPY", date(2024,3,1))
        strikes = sorted(q.strike for q in ch.quotes)
        self.assertEqual(strikes, [465.0, 470.0])

    def test_uses_vendor_bid_ask_verbatim(self):
        ch = self.src.get_chain("SPY", date(2024,3,1))
        q = [x for x in ch.quotes if x.strike == 470.0][0]
        self.assertAlmostEqual(q.bid, 5.00)
        self.assertAlmostEqual(q.ask, 5.10)

    def test_prefers_vendor_greeks_when_present(self):
        ch = self.src.get_chain("SPY", date(2024,3,1))
        q = [x for x in ch.quotes if x.strike == 470.0][0]
        self.assertAlmostEqual(q.delta, -0.30, places=4)
        self.assertAlmostEqual(q.iv, 0.16, places=4)

    def test_missing_date_returns_none(self):
        self.assertIsNone(self.src.get_chain("SPY", date(2024,3,2)))

    def test_index_is_reused_not_rebuilt(self):
        from vendor_data import VendorFileDataSource
        s2 = VendorFileDataSource("/nonexistent/path.csv", "SPY", preset="cboe",
                                  index_dir=os.path.join(self.tmp,"idx"),
                                  verbose=False)
        s2.prepare()      # must not touch the missing source file
        self.assertTrue(s2.trading_days(date(2024,1,1), date(2024,12,31)))

    def test_wrong_preset_fails_loudly(self):
        from vendor_data import VendorFileDataSource
        s = VendorFileDataSource(self.csv, "SPY", preset="orats",
                                 index_dir=os.path.join(self.tmp,"idx2"),
                                 verbose=False)
        with self.assertRaises(RuntimeError):
            s.prepare()


class TestNoArbitrageBound(unittest.TestCase):
    def test_bug_loss_cannot_exceed_width_minus_credit(self):
        """Shipped bug: the modelled bid/ask added a half-spread to each leg
        independently, so a deep-ITM 3-wide spread priced at $368 to close
        against a $300 structural ceiling -- a loss $60 worse than the position
        could possibly incur. Arbitrage forbids it; the model did not."""
        from engine import Backtest, BacktestConfig, FillModel
        data = SyntheticDataSource(start=date(2024,1,1), end=date(2024,12,31),
                                   s0=100.0, realized_vol=0.35, vrp=0.01,
                                   drift=-0.30, seed=4)   # steep decline -> deep ITM
        cfg = BacktestConfig(underlying="SYN", start=date(2024,1,1),
                             end=date(2024,12,31), starting_cash=25_000)
        bt = Backtest(data, PutCreditSpread(short_delta=0.30, wing_width=5,
                      target_dte=45, profit_take=0.50, stop_mult=2.0,
                      min_dte=21), cfg, FillModel(slippage_frac=1.0)).run()
        self.assertGreater(len(bt.closed), 5, "need losing trades to test the bound")
        for t in bt.closed:
            ml = t.position.max_loss
            if ml is None:
                continue
            fees = 2 * 0.05 * sum(l.qty for l in t.position.legs)
            self.assertGreaterEqual(
                t.pnl, -(ml + fees) - 0.01,
                f"loss ${t.pnl:.2f} exceeds structural max ${ml:.2f} + fees")


class TestNoStopConfig(unittest.TestCase):
    """`stop_mult=None` means hold to expiry with no stop. It is the config you
    need to compare against the CBOE PUT index, and it used to raise."""

    def _chain(self, spot=100.0, as_of=None, expiry=None):
        from data import OptionChain, OptionQuote
        as_of = as_of or date(2025, 6, 2)
        expiry = expiry or date(2025, 7, 2)
        # Distinct deltas, and strikes below the 30-delta one, or the wing has
        # nowhere to go and propose_entry correctly returns None.
        qs = []
        for k, d, px in ((80.0, -0.05, 0.15), (85.0, -0.12, 0.40),
                         (90.0, -0.30, 1.10), (95.0, -0.50, 2.60)):
            qs.append(OptionQuote(right="put", strike=k, expiry=expiry,
                                  bid=px, ask=px * 1.2, delta=d, iv=0.20))
        return OptionChain(as_of, "TEST", spot, qs)

    def test_bug_no_stop_mult_crashes(self):
        """stop_mult=None raised TypeError on -stop_mult; profit_take was
        guarded and stop_mult was not."""
        from strategies import PutCreditSpread
        s = PutCreditSpread(short_delta=0.30, wing_width=5, target_dte=30,
                            profit_take=None, stop_mult=None, min_dte=0)
        chain = self._chain()
        pos = s.propose_entry(chain, 0)
        self.assertIsNotNone(pos, "fixture should produce a position")
        try:
            s.manage(pos, chain)          # must not raise
        except TypeError as e:
            self.fail(f"stop_mult=None must mean 'no stop', not crash: {e}")

    def test_no_stop_still_honours_profit_take(self):
        """Disabling the stop must not disable the profit target."""
        from strategies import PutCreditSpread
        s = PutCreditSpread(short_delta=0.30, wing_width=5, target_dte=30,
                            profit_take=0.50, stop_mult=None, min_dte=0)
        pos = s.propose_entry(self._chain(), 0)
        self.assertIsNotNone(pos)
        s.manage(pos, self._chain())      # exercises both branches, no raise


class TestConcentrationCap(unittest.TestCase):
    """The portfolio cap alone permits full concentration in one name. The dev
    book held IWM 302/299p AND 303/300p simultaneously -- adjacent strikes, same
    expiry, same underlying. One gap takes both to max loss."""

    def _quotes(self):
        from data import OptionQuote
        exp = date(2026, 9, 18)
        short = OptionQuote(right="put", strike=300.0, expiry=exp, bid=1.00,
                            ask=1.10, delta=-0.30, iv=0.22, symbol="IWM260918P00300000")
        long = OptionQuote(right="put", strike=297.0, expiry=exp, bid=0.40,
                           ask=0.48, delta=-0.18, iv=0.24, symbol="IWM260918P00297000")
        return short, long

    def _limits(self, **kw):
        from execution import RiskLimits
        base = dict(account_equity=25_000.0, risk_per_trade_frac=0.05,
                    max_contracts=5, max_risk_per_trade=2_000.0,
                    max_open_risk=1_500.0, min_credit_per_contract=20.0,
                    max_spread_frac_of_credit=0.90)
        base.update(kw)
        return RiskLimits(**base)

    def test_bug_portfolio_cap_allows_full_concentration(self):
        """With 50% per-name cap, a third IWM spread must be refused once
        $750 of IWM risk is already open -- even though the $1,500 PORTFOLIO
        cap still shows $750 of room."""
        from execution import size_trade, RiskRefusal
        short, long = self._quotes()
        with self.assertRaises(RiskRefusal) as ctx:
            size_trade("TOKEN", "IWM", None, short, long,
                       self._limits(max_open_risk_per_underlying_frac=0.50),
                       open_risk=750.0, open_risk_this_underlying=750.0)
        self.assertIn("concentration", str(ctx.exception).lower())

    def test_other_underlying_still_allowed(self):
        """The cap is PER NAME: SPY must still size with $750 of IWM open."""
        from execution import size_trade
        short, long = self._quotes()
        t = size_trade("TOKEN", "SPY", None, short, long,
                       self._limits(max_open_risk_per_underlying_frac=0.50),
                       open_risk=750.0, open_risk_this_underlying=0.0)
        self.assertGreaterEqual(t.contracts, 1)

    def test_cap_defaults_are_backward_compatible(self):
        """Callers that do not pass open_risk_this_underlying must behave as
        before -- the new arg defaults to 0."""
        from execution import size_trade
        short, long = self._quotes()
        t = size_trade("TOKEN", "IWM", None, short, long,
                       self._limits(), open_risk=0.0)
        self.assertGreaterEqual(t.contracts, 1)


class TestOpenRiskDetection(unittest.TestCase):
    """open_risk() returned 0.0 for a book full of options, so max_open_risk
    never fired. Cause: alpaca-py's AssetClass enum stringifies as
    "AssetClass.US_OPTION" -- uppercase -- and the test was a case-sensitive
    `"option" in str(...)`."""

    class _AssetClass:
        """Mimics alpaca-py's enum: uppercase str(), lowercase .value."""
        value = "us_option"
        def __str__(self):
            return "AssetClass.US_OPTION"

    def _pos(self, symbol, qty):
        return SimpleNamespace(symbol=symbol, qty=str(qty),
                               asset_class=self._AssetClass())

    def _executor(self, positions):
        trading = SimpleNamespace(get_all_positions=lambda: positions)
        return ex.Executor(trading, None, ex.RiskLimits())

    def test_bug_open_risk_blind_to_uppercase_asset_class(self):
        """Three short contracts must not read as zero risk."""
        e = self._executor([self._pos("IWM260817P00302000", -1),
                            self._pos("IWM260817P00303000", -1),
                            self._pos("SPY260817P00775000", -1)])
        self.assertGreater(e.open_risk(), 0.0,
                           "short options must register as open risk")

    def test_open_risk_filters_by_underlying(self):
        e = self._executor([self._pos("IWM260817P00302000", -1),
                            self._pos("IWM260817P00303000", -1),
                            self._pos("SPY260817P00775000", -1)])
        self.assertAlmostEqual(e.open_risk("IWM"), 2 * e.open_risk("SPY"))
        self.assertAlmostEqual(e.open_risk("IWM") + e.open_risk("SPY"),
                               e.open_risk())
        self.assertEqual(e.open_risk("QQQ"), 0.0)

    def test_long_legs_do_not_add_risk(self):
        """Only SHORT options carry the defined-risk exposure."""
        e = self._executor([self._pos("IWM260817P00299000", 1),
                            self._pos("IWM260817P00300000", 1)])
        self.assertEqual(e.open_risk(), 0.0)

    def test_unreachable_broker_reports_infinite_risk(self):
        """If exposure cannot be verified, behave as if fully allocated."""
        def boom():
            raise RuntimeError("broker down")
        e = ex.Executor(SimpleNamespace(get_all_positions=boom), None, ex.RiskLimits())
        self.assertEqual(e.open_risk(), float("inf"))


class TestAutoAcceptInterlock(unittest.TestCase):
    """auto_accept is a paper-only convenience. The hazard is not auto-accept
    itself -- it is auto-accept surviving a later flip to live_money, leaving an
    unattended trader spending real money. The two must not both take effect."""

    def _flag(self, cfg, name):
        return ex.flag_on(cfg, name)

    def test_flag_parses_on_variants(self):
        for v in ("on", "ON", "true", "1", "yes"):
            self.assertTrue(self._flag({"auto_accept": v}, "auto_accept"), v)
        for v in ("off", "false", "0", "no", "", "maybe"):
            self.assertFalse(self._flag({"auto_accept": v}, "auto_accept"), v)

    def test_bug_auto_accept_must_not_apply_when_live(self):
        """The interlock: auto AND live must resolve to NOT auto."""
        for auto_v, live_v, expect in (("on", "off", True),
                                       ("on", "on", False),
                                       ("off", "on", False),
                                       ("off", "off", False)):
            cfg = {"auto_accept": auto_v, "live_money": live_v}
            effective = ex.auto_accept_effective(cfg)
            self.assertEqual(effective, expect,
                             f"auto={auto_v} live={live_v} -> {effective}")

    def test_synthesized_reply_is_a_valid_confirmation(self):
        """The auto path builds Reply objects the submit path already trusts."""
        r = nt.Reply(token="A3F", confirmed=True, contracts=None,
                     raw="<auto_accept>",
                     received_at=datetime.now(timezone.utc))
        self.assertTrue(r.confirmed)
        self.assertIsNone(r.contracts, "must use the proposed size, not invent one")
        self.assertEqual(r.token, "A3F")


class TestBreakevenArithmetic(unittest.TestCase):
    """The clearest statement of what this strategy is: a 30-delta short implies
    a 70% win rate, so a FAIR payoff ratio is 0.30/0.70 = 0.43. The backtest
    measures 0.42 across all twelve exit-rule variants. Fair value, no premium."""

    def test_breakeven_matches_hand_arithmetic(self):
        # $50 credit on a 3-wide ($300) spread, take 50% / stop 2x
        #   win  = 0.50 * 50 = 25 ; loss = min(2*50, 250) = 100
        #   be   = 100 / 125 = 0.80
        self.assertAlmostEqual(
            ex.breakeven_win_rate(50.0, 250.0, 0.50, 2.0), 0.80, places=4)

    def test_no_stop_uses_full_max_loss(self):
        # loss = max_loss = 250 ; win = 0.50*50 = 25 -> 250/275
        self.assertAlmostEqual(
            ex.breakeven_win_rate(50.0, 250.0, 0.50, None), 250/275, places=4)

    def test_no_rules_at_all_is_width_over_width(self):
        # win = full credit 50, loss = 250 -> 250/300
        self.assertAlmostEqual(
            ex.breakeven_win_rate(50.0, 250.0, None, None), 250/300, places=4)

    def test_implied_win_rate_from_delta(self):
        self.assertAlmostEqual(ex.implied_win_rate(-0.30), 0.70, places=6)
        self.assertAlmostEqual(ex.implied_win_rate(0.30), 0.70, places=6)
        self.assertAlmostEqual(ex.implied_win_rate(-0.50), 0.50, places=6)

    def test_fair_payoff_ratio_is_delta_over_one_minus_delta(self):
        """A fair 30-delta short pays 0.43; the backtest realises 0.42."""
        d = 0.30
        fair = d / (1 - d)
        self.assertAlmostEqual(fair, 0.4286, places=3)
        measured = 0.42
        self.assertLess(abs(fair - measured), 0.02,
                        "measured payoff should sit at fair value, not above it")

    def test_breakeven_check_does_not_refuse(self):
        """It is informational. Gating on the nominal number would refuse every
        configuration tested, including the one that backtests best."""
        import inspect
        src = inspect.getsource(ex.size_trade)
        i = src.find("breakeven_win_rate")
        self.assertGreater(i, 0)
        window = src[i:i + 400]
        self.assertNotIn("raise RiskRefusal", window,
                         "break-even must annotate, not refuse")



class TestGammaExposure(unittest.TestCase):
    """NGE = [sum_calls(gamma*OI*100*S) - sum_puts(...)] / market_value.
    The sign convention encodes the standing assumption of this literature:
    dealers are long every call and short every put."""

    def _snap(self, **kw):
        import gamma as gm
        base = dict(as_of="2026-08-24T00:00:00Z", underlying="SPY", spot=700.0,
                    call_gamma_dollars=0.0, put_gamma_dollars=0.0,
                    net_gamma_dollars=0.0, nge=0.0, contracts_used=0,
                    contracts_missing_oi=0, oi_as_of="2026-08-20",
                    market_value=7e11)
        base.update(kw)
        return gm.GammaSnapshot(**base)

    def test_more_put_gamma_means_dealers_short(self):
        """Puts dominating -> net negative -> the regime the signal needs."""
        s = self._snap(call_gamma_dollars=1.0e9, put_gamma_dollars=2.0e9,
                       net_gamma_dollars=-1.0e9)
        self.assertTrue(s.dealers_short_gamma)

    def test_more_call_gamma_means_dealers_long(self):
        s = self._snap(call_gamma_dollars=2.0e9, put_gamma_dollars=1.0e9,
                       net_gamma_dollars=1.0e9)
        self.assertFalse(s.dealers_short_gamma)

    def test_zero_is_not_short(self):
        """Baltussen split on NGE < 0, so exactly zero belongs to the
        no-effect bucket."""
        self.assertFalse(self._snap(net_gamma_dollars=0.0).dealers_short_gamma)


class TestIntradaySignals(unittest.TestCase):
    """Signal definitions, and the exit price that decided the result."""

    def _day(self, prev=100.0, o930=100.0, c1000=101.0,
             c1500=102.0, c1530=103.0, close=104.0):
        import intraday as I
        return I.Day(date(2026, 8, 24), prev, o930, c1000, c1500, c1530, close)

    def test_signal_definitions(self):
        d = self._day()
        self.assertAlmostEqual(d.signal("onfh"), (101.0-100.0)/100.0)
        self.assertAlmostEqual(d.signal("rod"),  (103.0-100.0)/100.0)
        self.assertAlmostEqual(d.signal("r12"),  (103.0-102.0)/102.0)

    def test_bug_target_uses_auction_print_not_1559_bar(self):
        """The strategy exits in the closing auction. Its price is the DAILY
        bar's close, which differs from the 15:59 minute bar by 0.1-0.6 bp --
        up to a fifth of a ~2.7 bp edge."""
        d = self._day(c1530=103.0, close=104.0)
        self.assertAlmostEqual(d.target, (104.0-103.0)/103.0)

    def test_unknown_signal_raises(self):
        with self.assertRaises(ValueError):
            self._day().signal("not_a_signal")

    def test_cost_is_charged_per_trade(self):
        """A signal with zero predictive content must lose exactly the cost."""
        import intraday as I
        days = [self._day(c1530=100.0, close=100.0) for _ in range(40)]
        r = I.evaluate(days, "r12", cost_bp=0.30)
        self.assertAlmostEqual(r["net_bp"], -0.30, places=6)



class TestOpenRiskPairing(unittest.TestCase):
    """Legs must pair into spreads. Assuming every short is 5-wide overstated a
    3-wide book by 67% and would park the system at its ceiling with 40% of the
    budget genuinely free."""

    class _AC:
        value = "us_option"
        def __str__(self): return "AssetClass.US_OPTION"

    def _p(self, sym, qty):
        return SimpleNamespace(symbol=sym, qty=str(qty), asset_class=self._AC())

    def _e(self, positions):
        return ex.Executor(SimpleNamespace(get_all_positions=lambda: positions),
                           None, ex.RiskLimits())

    def test_three_wide_spread_is_three_hundred(self):
        e = self._e([self._p("SPY260817P00775000", -1),
                     self._p("SPY260817P00772000", 1)])
        self.assertAlmostEqual(e.open_risk(), 300.0)

    def test_two_spreads_same_name_sum(self):
        """IWM 303/300 and 302/299 -> $600, not one spread's worth."""
        e = self._e([self._p("IWM260817P00303000", -1),
                     self._p("IWM260817P00300000", 1),
                     self._p("IWM260817P00302000", -1),
                     self._p("IWM260817P00299000", 1)])
        self.assertAlmostEqual(e.open_risk("IWM"), 600.0)

    def test_naked_short_still_assumed_wide(self):
        """No long to pair with -> fall back to the conservative estimate."""
        e = self._e([self._p("SPY260817P00775000", -1)])
        self.assertAlmostEqual(e.open_risk(), 500.0)

    def test_portfolio_equals_sum_of_names(self):
        e = self._e([self._p("SPY260817P00775000", -1),
                     self._p("SPY260817P00772000", 1),
                     self._p("IWM260817P00303000", -1),
                     self._p("IWM260817P00300000", 1)])
        self.assertAlmostEqual(e.open_risk(),
                               e.open_risk("SPY") + e.open_risk("IWM"))
        self.assertAlmostEqual(e.open_risk(), 600.0)

    def test_different_expiries_do_not_pair(self):
        """A short and a long in different cycles are not a spread."""
        e = self._e([self._p("SPY260817P00775000", -1),
                     self._p("SPY260918P00772000", 1)])
        self.assertAlmostEqual(e.open_risk(), 500.0)


# --------------------------------------------------------------------------- #
class TestExitSettlement(unittest.TestCase):
    """A close is not a close until it fills.

    The shipped bug: manage() deleted the tracking record the instant
    submit_order returned. A DAY limit that never filled left the spread open
    at the broker with nothing tracking it, so no later cycle would ever try to
    exit it again. One IWM 296/293 spread ran to expiry that way, unmanaged and
    absent from open_risk.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.cwd = os.getcwd(); os.chdir(self.tmp)
        storage.use(positions=storage.FilePositionStore())
        self.sent = []
        self.notifier = SimpleNamespace(send=lambda m: self.sent.append(m))
        self.pos = {"underlying": "IWM", "short_occ": "IWM260831P00296000",
                    "long_occ": "IWM260831P00293000", "short_strike": 296,
                    "long_strike": 293, "contracts": 1, "expiry": "2026-08-31",
                    "entry_credit_per_contract": 48, "token": "M8P",
                    "closing": {"order_id": "o1", "reason": "profit target 50%",
                                "submitted_at": "2026-08-28T15:30:02+00:00",
                                "limit_price": 0.25}}

    def tearDown(self):
        os.chdir(self.cwd)

    def _clients(self, status):
        return {"trading": SimpleNamespace(
            get_order_by_id=lambda oid: SimpleNamespace(status=status))}

    def test_bug_unfilled_close_keeps_the_position_tracked(self):
        import handlers
        storage.POSITIONS.put("M8P", self.pos)
        positions = storage.POSITIONS.all()
        acted = []
        flight = handlers._settle_closing(positions, self._clients("OrderStatus.NEW"),
                                          self.notifier, acted)
        # Still working: keep it, and do not submit a second close this cycle.
        self.assertIn("M8P", flight)
        self.assertIn("M8P", storage.POSITIONS.all())

    def test_bug_expired_close_clears_marker_and_retries(self):
        import handlers
        storage.POSITIONS.put("M8P", self.pos)
        positions = storage.POSITIONS.all()
        acted = []
        flight = handlers._settle_closing(positions,
                                          self._clients("OrderStatus.EXPIRED"),
                                          self.notifier, acted)
        # The order died unfilled. The position must survive, with the marker
        # cleared so the normal exit logic re-prices and resubmits.
        self.assertNotIn("M8P", flight)
        rec = storage.POSITIONS.all().get("M8P")
        self.assertIsNotNone(rec)
        self.assertNotIn("closing", rec)
        self.assertTrue(any("did NOT fill" in m for m in self.sent))

    def test_filled_close_removes_the_position(self):
        import handlers
        storage.POSITIONS.put("M8P", self.pos)
        positions = storage.POSITIONS.all()
        acted = []
        handlers._settle_closing(positions, self._clients("OrderStatus.FILLED"),
                                 self.notifier, acted)
        self.assertEqual(storage.POSITIONS.all(), {})
        self.assertIn("close confirmed", acted[0]["result"])

    def test_reconcile_flags_a_broker_position_we_do_not_track(self):
        import handlers
        held = [SimpleNamespace(symbol="IWM260831P00296000", asset_class="us_option"),
                SimpleNamespace(symbol="IWM260831P00293000", asset_class="us_option")]
        clients = {"trading": SimpleNamespace(get_all_positions=lambda: held)}
        out = handlers._reconcile(clients, {}, self.notifier)
        self.assertEqual(len(out["untracked"]), 2)
        self.assertTrue(any("UNTRACKED" in m for m in self.sent))

    def test_reconcile_silent_when_everything_is_tracked(self):
        import handlers
        held = [SimpleNamespace(symbol="IWM260831P00296000", asset_class="us_option"),
                SimpleNamespace(symbol="IWM260831P00293000", asset_class="us_option")]
        clients = {"trading": SimpleNamespace(get_all_positions=lambda: held)}
        out = handlers._reconcile(clients, {"M8P": self.pos}, self.notifier)
        self.assertEqual(out["untracked"], [])
        self.assertEqual(self.sent, [])

    def test_reconcile_alerts_once_not_every_cycle(self):
        """manage runs ~90 times a day. An unthrottled alert trains you to
        ignore the one push that means money is moving unwatched."""
        import handlers

        class _Cache:
            def __init__(self): self.v = {}
            def get(self, k): return self.v.get(k)
            def put(self, k, val, ttl_seconds=0): self.v[k] = val

        cache = _Cache()
        held = [SimpleNamespace(symbol="IWM260831P00296000", asset_class="us_option")]
        clients = {"trading": SimpleNamespace(get_all_positions=lambda: held)}
        for _ in range(5):
            handlers._reconcile(clients, {}, self.notifier, cache)
        self.assertEqual(len(self.sent), 1, "should alert once, not once per cycle")

    def test_reconcile_realerts_when_the_set_changes(self):
        import handlers

        class _Cache:
            def __init__(self): self.v = {}
            def get(self, k): return self.v.get(k)
            def put(self, k, val, ttl_seconds=0): self.v[k] = val

        cache = _Cache()
        one = [SimpleNamespace(symbol="IWM260831P00296000", asset_class="us_option")]
        two = one + [SimpleNamespace(symbol="SPY260831P00763000",
                                     asset_class="us_option")]
        handlers._reconcile({"trading": SimpleNamespace(get_all_positions=lambda: one)},
                            {}, self.notifier, cache)
        handlers._reconcile({"trading": SimpleNamespace(get_all_positions=lambda: two)},
                            {}, self.notifier, cache)
        self.assertEqual(len(self.sent), 2, "a NEW orphan must break through")


    def test_bug_assigned_shares_are_flagged_too(self):
        """A short put assigned at expiry becomes an EQUITY position. Filtering
        reconcile on asset_class=='option' made the most expensive orphan --
        100 shares of unmanaged stock -- the one case it could not see."""
        import handlers
        held = [SimpleNamespace(symbol="IWM", asset_class="us_equity", qty="100")]
        clients = {"trading": SimpleNamespace(get_all_positions=lambda: held)}
        out = handlers._reconcile(clients, {}, self.notifier)
        self.assertEqual(out["untracked"], ["IWM"])
        self.assertTrue(any("UNTRACKED" in m for m in self.sent))

    def test_tracked_option_legs_still_do_not_alert(self):
        """Widening the filter must not make the normal case noisy."""
        import handlers
        held = [SimpleNamespace(symbol="IWM260831P00296000", asset_class="us_option"),
                SimpleNamespace(symbol="IWM260831P00293000", asset_class="us_option")]
        clients = {"trading": SimpleNamespace(get_all_positions=lambda: held)}
        out = handlers._reconcile(clients, {"M8P": self.pos}, self.notifier)
        self.assertEqual(out["untracked"], [])
        self.assertEqual(self.sent, [])


    def test_bug_unfilled_close_books_no_realised_pnl(self):
        """CLOSE-M9B booked +$21 at submit, then expired unfilled at the bell.
        That phantom profit flattered the day and freed daily capital budget
        that had never actually come back."""
        import handlers
        storage.POSITIONS.put("M9B", self.pos)
        positions = storage.POSITIONS.all()
        handlers._settle_closing(positions,
                                 self._clients("OrderStatus.EXPIRED"),
                                 self.notifier, [])
        booked = sum(float(r.get("realized_pnl") or 0)
                     for r in storage.JOURNAL.todays_submitted())
        self.assertEqual(booked, 0.0, "an unfilled close must book nothing")
        self.assertIn("M9B", storage.POSITIONS.all())

    def test_realised_pnl_uses_the_fill_not_the_limit(self):
        """The limit is what we asked for; the fill is what happened."""
        import handlers
        storage.POSITIONS.put("M9B", self.pos)          # entry credit $48
        positions = storage.POSITIONS.all()
        filled = {"trading": SimpleNamespace(
            get_order_by_id=lambda oid: SimpleNamespace(
                status="OrderStatus.FILLED", filled_avg_price="0.10"))}
        acted = []
        handlers._settle_closing(positions, filled, self.notifier, acted)
        # limit was 0.25 -> would book +$23; fill at 0.10 -> +$38.
        self.assertAlmostEqual(acted[0]["realized_pnl"], 38.0)
        booked = sum(float(r.get("realized_pnl") or 0)
                     for r in storage.JOURNAL.todays_submitted())
        self.assertAlmostEqual(booked, 38.0)
        self.assertEqual(storage.POSITIONS.all(), {})


# --------------------------------------------------------------------------- #
class TestOptIntParsing(unittest.TestCase):
    """A cap that parses to the wrong thing is a safety control that lies."""

    def test_spellings_that_mean_no_cap(self):
        for v in (None, "", " ", "none", "NONE", "off", "unlimited", "null",
                  "0", 0, "-3"):
            self.assertIsNone(ex.opt_int(v), f"{v!r} should mean no cap")

    def test_numeric_caps_survive(self):
        self.assertEqual(ex.opt_int("12"), 12)
        self.assertEqual(ex.opt_int(8), 8)
        self.assertEqual(ex.opt_int("8.0"), 8)

    def test_garbage_falls_back_to_the_default(self):
        self.assertEqual(ex.opt_int("banana", 5), 5)
        self.assertIsNone(ex.opt_int("banana"))


# --------------------------------------------------------------------------- #
class TestExpiryFloor(unittest.TestCase):
    """Entry must never select a DTE the exit rules reject on sight.

    Shipped bug (2026-09-03): nearest_expiry went purely by distance to the
    target, so with target_dte=3 and Labor Day removing the Monday expiry it
    chose Friday at 1 DTE. manage() closes at dte <= min_dte = 1, so three
    positions opened at 13:35 were closed at 13:40 -- three round trips, five
    minutes held, and the rest of the session refused.
    """

    AS_OF = date(2026, 9, 3)          # a Thursday

    def _chain(self, expiries):
        from data import OptionChain
        qs = []
        for e in expiries:
            for k, d, px in ((289.0, -0.12, 0.40), (292.0, -0.30, 1.10),
                             (294.0, -0.40, 1.80), (296.0, -0.50, 2.60)):
                qs.append(OptionQuote(right="put", strike=k, expiry=e,
                                      bid=px, ask=px * 1.2, delta=d, iv=0.20))
        return OptionChain(self.AS_OF, "IWM", 293.0, qs)

    def test_bug_does_not_pick_the_dte_the_exit_would_close(self):
        # Friday is 1 DTE and nearest the 3-DTE target by raw distance; the
        # next listing is 6 DTE. With min_dte=1, Friday must be excluded.
        chain = self._chain([date(2026, 9, 4), date(2026, 9, 9)])
        self.assertEqual(chain.nearest_expiry(3, 0), date(2026, 9, 4),
                         "unfloored selection reproduces the bug")
        self.assertEqual(chain.nearest_expiry(3, 1), date(2026, 9, 9),
                         "floored selection must skip the 1-DTE expiry")

    def test_same_day_expiry_is_never_selected(self):
        chain = self._chain([self.AS_OF, date(2026, 9, 11)])
        self.assertEqual(chain.nearest_expiry(3, 1), date(2026, 9, 11))

    def test_returns_none_when_everything_is_below_the_floor(self):
        """No tradeable expiry is a refusal, not a bad pick."""
        chain = self._chain([date(2026, 9, 4)])
        self.assertIsNone(chain.nearest_expiry(3, 1))

    def test_floor_of_zero_preserves_old_behaviour_for_other_callers(self):
        """events.atm_iv asks for ~30 DTE with no floor; it must be unaffected."""
        chain = self._chain([date(2026, 9, 4), date(2026, 10, 2)])
        self.assertEqual(chain.nearest_expiry(30), date(2026, 10, 2))

    def test_strategy_entry_respects_its_own_min_dte(self):
        from strategies import PutCreditSpread
        chain = self._chain([date(2026, 9, 4), date(2026, 9, 9)])
        s = PutCreditSpread(short_delta=0.30, wing_width=3, target_dte=3,
                            profit_take=0.50, stop_mult=2.0, min_dte=1)
        pos = s.propose_entry(chain, 0)
        self.assertIsNotNone(pos)
        dte = (pos.legs[0].expiry - self.AS_OF).days
        self.assertGreater(dte, 1, "entry picked a DTE the exit closes at once")


# --------------------------------------------------------------------------- #
class TestSparseChainWidening(unittest.TestCase):
    """leg_at_offset takes the NEAREST listed strike, so a sparse chain widens
    the spread past wing_width. That is safe -- max_loss uses the real strikes
    and the budget refuses what it cannot cover -- but on 2026-09-03 it surfaced
    as "wing_width=3, so how is defined risk $308?", which reads like impossible
    arithmetic. The refusal must name the width it actually got.
    """

    def _chain(self, strikes):
        from data import OptionChain
        exp = date(2026, 9, 11)
        qs = [OptionQuote(right="put", strike=k, expiry=exp, bid=1.0, ask=1.1,
                          delta=-0.30, iv=0.20, symbol=f"IWM{k:.0f}")
              for k in strikes]
        return OptionChain(date(2026, 9, 3), "IWM", 293.0, qs)

    def test_missing_strike_widens_the_spread(self):
        """No strike 3 below -> the wing lands 4 below, not nowhere."""
        chain = self._chain([289.0, 290.0, 293.0])       # 290 present
        leg = chain.leg_at_offset(date(2026, 9, 11), "put", 293.0, 3)
        self.assertEqual(leg.strike, 290.0)

        sparse = self._chain([289.0, 293.0])             # 290 absent
        leg = sparse.leg_at_offset(date(2026, 9, 11), "put", 293.0, 3)
        self.assertEqual(leg.strike, 289.0, "falls back to the nearest listed")

    def test_refusal_names_the_realised_width(self):
        """The message must explain itself without a log dive."""
        L = ex.RiskLimits(max_risk_per_trade=300.0, account_equity=25_000.0,
                          risk_per_trade_frac=0.02)
        short_q = q(293, 5.00, 5.10, sym="S")
        long_q = q(289, 4.05, 4.15, sym="L")             # 4 wide, not 3
        with self.assertRaises(ex.RiskRefusal) as cm:
            ex.size_trade("W1", "IWM", None, short_q, long_q, L)
        msg = str(cm.exception)
        self.assertIn("$4 wide", msg)
        self.assertIn("293/289p", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
