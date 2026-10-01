"""Wright's law with a floor and an anchor, priced over fractional units.

Wright's law (parent ``wright1936_airplane_cost``) gives the ``n``th unit's
cost as ``T_1 n^b``, with ``b = log2(r)`` and ``r`` the learning rate, the
share of unit cost kept each time cumulative production doubles.  The growth
cost model (ADR 0037) adds two things to it:

- a **floor**, the materials and labour no volume removes:
  ``T(n) = floor + (T_1 - floor) (n / A)^b``;
- an **anchor** ``A``, the cumulative count up to which the price holds flat.
  Plates and chambers anchor at their first unit.  The steering package holds
  its at-volume price until 100 000 are built (ASKS H6).

The program buys fractional launch units, so a batch's cost is an integral.
It is the **midpoint** integral, ``int_{N0+1/2}^{N0+k+1/2} T(x) dx``, which
gives unit ``n`` the price over ``[n - 1/2, n + 1/2]``.  On an 80% curve it
runs slightly below the discrete sum: 3% on the first unit alone, under 2% from
two units and under 1% from six.  The scratch model
integrated from zero instead, which charges every program about one extra first
unit (+47% on unit 1 at 80%).
"""

import math
from dataclasses import dataclass
from typing import Optional

_HALF = 0.5


@dataclass(frozen=True)
class LearningCurve:
    """A unit price that falls with cumulative production.

    Attributes:
        first: Price of each unit up to the anchor.
        floor: Price the curve approaches and never crosses.
        rate: Share of the price above the floor kept per doubling; 1 is flat.
        anchor: Cumulative count at which learning starts.
    """

    first: float
    floor: float = 0.0
    rate: float = 1.0
    anchor: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.rate <= 1.0:
            raise ValueError("the learning rate must lie in (0, 1]")
        if self.floor > self.first:
            raise ValueError("the floor must not exceed the first price")
        if self.anchor < _HALF:
            raise ValueError("the anchor must be at least half a unit")

    @classmethod
    def flat(cls, price: float) -> "LearningCurve":
        """A price that never moves.

        Args:
            price: The price of every unit.

        Returns:
            The curve.
        """
        return cls(price, price, 1.0)

    @property
    def _exponent(self) -> float:
        return math.log2(self.rate)

    def unit(self, n: float) -> float:
        """Price of unit ``n``, counting from one.

        Args:
            n: Cumulative unit number.

        Returns:
            The price.
        """
        span = self.first - self.floor
        return float(
            self.floor + span * (max(n, self.anchor) / self.anchor) ** self._exponent
        )

    def _area(self, x: float) -> float:
        """``int_{1/2}^{x} (T(s) - floor) ds``."""
        span = self.first - self.floor
        if x <= self.anchor:
            return span * (x - _HALF)
        flat = span * (self.anchor - _HALF)
        b = self._exponent
        ratio = x / self.anchor
        if math.isclose(b, -1.0):
            tail = self.anchor * math.log(ratio)
        else:
            tail = self.anchor * (ratio ** (1.0 + b) - 1.0) / (1.0 + b)
        return flat + span * tail

    def batch_cost(self, built: float, count: float) -> float:
        """Price of ``count`` more units after ``built`` already exist.

        Args:
            built: Units bought so far.
            count: Units in this batch; fractions allowed.

        Returns:
            The batch's price.
        """
        if count <= 0.0:
            return 0.0
        start, end = built + _HALF, built + count + _HALF
        return self.floor * count + self._area(end) - self._area(start)

    def units_to(self, price: float) -> Optional[float]:
        """Cumulative count at which the unit price first falls to ``price``.

        Args:
            price: The target unit price.

        Returns:
            The count, or None if the floor never lets it get there.
        """
        if price >= self.first:
            return 1.0
        if price <= self.floor or self.rate == 1.0:
            return None
        share = (price - self.floor) / (self.first - self.floor)
        return float(self.anchor * share ** (1.0 / self._exponent))
