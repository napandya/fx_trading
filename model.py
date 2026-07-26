from openmic.projects.fx_algo_replay.market_state import MarketState
from openmic.projects.fx_algo_replay.pricing_strategy import PricingStrategy
from openmic.projects.fx_algo_replay.quote_action import QuoteAction

# --- extra std-lib helpers (the 3 required imports above are unchanged) ---
from collections import deque


class SubmittedPricingStrategy(PricingStrategy):
    """
    Adaptive EURUSD market maker.

    Every tick it starts from the market-neutral quote and adjusts four things:

      1. SPREAD   - base offset derived from the live market spread, widened when
                    recent volatility rises or recent fills look toxic.
      2. SKEW     - shifts bid/ask to lean against current inventory so we get
                    pulled back toward flat.
      3. SIZE     - quotes larger on the side that reduces inventory, smaller on
                    the side that grows it; hard guardrails stop runaway positions.
      4. END-OF-WEEK - skews harder on Fridays to reduce the forced-liquidation
                    penalty from carrying inventory over the weekend.

    All the real logic is wrapped in a try/except that falls back to a safe fixed
    quote, so a bad tick can never crash the replay.
    """

    PIP = 0.0001  # EURUSD pip size

    def __init__(self) -> None:
        super().__init__()

        # ---- fallback / baseline quote (used if adaptive logic ever errors) ----
        self.bid_offset_pips = 0.2
        self.ask_offset_pips = 0.2
        self.bid_size_m = 5.0
        self.ask_size_m = 5.0

        # ---------------------- TUNABLE PARAMETERS ----------------------
        # Spread ---------------------------------------------------------
        # We quote at base_offset_ratio * (market_half_spread). 1.0 = exactly at
        # the neutral bid/ask (captures full spread, still max fill prob).
        # < 1.0 leans slightly inside neutral to win a touch more market share.
        self.base_offset_ratio = 0.90
        self.min_offset_pips = 0.05
        self.max_offset_pips = 2.50

        # Size -----------------------------------------------------------
        self.base_size_m = 6.0
        self.min_size_m = 0.5
        self.max_size_m = 12.0
        self.size_skew = 0.60          # how strongly size leans with inventory

        # Inventory ------------------------------------------------------
        self.inv_soft_limit_m = 25.0   # skew scales to full strength here
        self.inv_hard_limit_m = 50.0   # beyond this we choke the growing side
        self.skew_strength_pips = 0.35 # max price skew (pips) at the soft limit

        # Volatility widening -------------------------------------------
        self.vol_window = 20
        self.vol_coef = 1.20           # pips of widening per pip of tick-vol
        self.max_vol_widen_pips = 1.00

        # Adverse-selection (mark-out) widening -------------------------
        self.toxicity_alpha = 0.10     # EMA smoothing on the mark-out signal
        self.toxicity_widen = 0.50     # pips widening per unit of toxicity
        self.max_tox_widen_pips = 0.80

        # End-of-week flattening ----------------------------------------
        self.friday_skew_mult = 1.80   # skew harder on Fridays

        # ---------------------------- STATE ----------------------------
        self._mids = deque(maxlen=self.vol_window)  # recent mids for vol estimate
        self._toxicity = 0.0                        # EMA of adverse mark-out (pips)

    # ------------------------------------------------------------------
    def quote(self, state: MarketState) -> QuoteAction:
        try:
            return self._quote_impl(state)
        except Exception:
            # Never let a single bad tick crash the run — fall back to baseline.
            return QuoteAction(
                bid_offset_pips=self.bid_offset_pips,
                ask_offset_pips=self.ask_offset_pips,
                bid_size_m=self.bid_size_m,
                ask_size_m=self.ask_size_m,
            )

    # ------------------------------------------------------------------
    def _quote_impl(self, state: MarketState) -> QuoteAction:
        mid = float(state.market_neutral_mid)
        mkt_spread_pips = float(state.market_neutral_spread_pip)
        inventory = float(getattr(state, "inventory_eur_m", 0.0) or 0.0)

        # --- update rolling market signals -----------------------------
        self._update_toxicity(state, mid)
        self._mids.append(mid)
        vol_pips = self._recent_vol_pips()

        # --- 1. BASE SPREAD --------------------------------------------
        half_spread = max(mkt_spread_pips, 0.0) / 2.0
        base_offset = self.base_offset_ratio * half_spread

        # widen for volatility and for recently toxic flow
        vol_widen = min(self.vol_coef * vol_pips, self.max_vol_widen_pips)
        tox_widen = min(self.toxicity_widen * max(self._toxicity, 0.0),
                        self.max_tox_widen_pips)
        base_offset += vol_widen + tox_widen

        # --- 2. INVENTORY SKEW -----------------------------------------
        inv_ratio = self._clip(inventory / self.inv_soft_limit_m, -1.0, 1.0)

        skew_strength = self.skew_strength_pips
        if str(getattr(state, "weekday", "")).lower().startswith("fri"):
            skew_strength *= self.friday_skew_mult  # push toward flat before close

        skew_pips = skew_strength * inv_ratio
        # long inventory (inv_ratio>0): raise bid offset (less attractive),
        # lower ask offset (more attractive) -> encourages us to sell it down.
        bid_offset = base_offset + skew_pips
        ask_offset = base_offset - skew_pips

        bid_offset = self._clip(bid_offset, self.min_offset_pips, self.max_offset_pips)
        ask_offset = self._clip(ask_offset, self.min_offset_pips, self.max_offset_pips)

        # --- 3. SIZE ---------------------------------------------------
        # shrink the side that grows inventory, grow the side that reduces it.
        bid_size = self.base_size_m * (1.0 - self.size_skew * inv_ratio)
        ask_size = self.base_size_m * (1.0 + self.size_skew * inv_ratio)

        # hard guardrails: if we're past the hard limit, choke the growing side.
        if inventory > self.inv_hard_limit_m:
            bid_size = self.min_size_m          # stop buying more EUR
        elif inventory < -self.inv_hard_limit_m:
            ask_size = self.min_size_m          # stop selling more EUR

        bid_size = self._clip(bid_size, self.min_size_m, self.max_size_m)
        ask_size = self._clip(ask_size, self.min_size_m, self.max_size_m)

        return QuoteAction(
            bid_offset_pips=bid_offset,
            ask_offset_pips=ask_offset,
            bid_size_m=bid_size,
            ask_size_m=ask_size,
        )

    # ------------------------------------------------------------------
    def _recent_vol_pips(self) -> float:
        """Std-dev of consecutive mid changes, expressed in pips."""
        n = len(self._mids)
        if n < 3:
            return 0.0
        mids = list(self._mids)
        diffs = [(mids[i] - mids[i - 1]) / self.PIP for i in range(1, n)]
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        return var ** 0.5

    # ------------------------------------------------------------------
    def _update_toxicity(self, state: MarketState, current_mid: float) -> None:
        """
        Approximate adverse selection via a one-step mark-out on previous fills.

        For each fill from last step, compare the mid then vs. the mid now:
          - BUY fill  (we bought EUR): favorable if mid rose.
          - SELL fill (we sold  EUR): favorable if mid fell.
        Adverse moves push toxicity up, which widens our spread next ticks.
        """
        fills = getattr(state, "previous_fills", None)
        if not fills:
            # slowly relax toxicity when nothing is trading against us
            self._toxicity *= (1.0 - self.toxicity_alpha)
            return

        for fill in fills:
            try:
                fill_mid = float(getattr(fill, "market_neutral_mid", current_mid))
                side = str(getattr(fill, "side", "")).upper()
                move_pips = (current_mid - fill_mid) / self.PIP
                if "SELL" in side:      # we sold EUR -> want mid to fall
                    markout = -move_pips
                else:                    # we bought EUR -> want mid to rise
                    markout = move_pips
                adverse = max(-markout, 0.0)  # only adverse moves raise toxicity
                self._toxicity += self.toxicity_alpha * (adverse - self._toxicity)
            except Exception:
                continue

    # ------------------------------------------------------------------
    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x