"""Configurable rule set.  Defaults follow the QQ-游戏 / 长沙 conventions.

三打哈 rules differ between regions and platforms; every point where the
common sources disagree is a field here (see docs/RULES.zh.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Rules:
    # --- bidding ---
    max_bid: int = 80          # highest (easiest) opening bid
    min_bid: int = 5           # lowest possible bid
    bid_step: int = 5
    # A player holding >= force_changzhu 常主 (王/7/2) may not pass while the
    # standing bid belongs to someone else and is above force_bid.
    force_changzhu: int = 7
    force_bid: int = 70
    force_enabled: bool = True
    # --- banker ---
    allow_no_trump: bool = True     # 无主
    allow_surrender: bool = True    # 投降 after seeing 底牌
    surrender_threshold: int = 55   # bid >= this: surrender counts as 下庄, else 小倒
    # --- play ---
    allow_trump_throw: bool = True  # 主牌甩牌 (side-suit 甩 is never allowed in this version)
    # --- 扣底 multipliers ---
    koudi_single: int = 1
    koudi_pair: int = 2
    koudi_per_tractor_pair: int = 2   # 2 pairs -> x4, 3 pairs -> x6 ...
    koudi_throw_per_card: bool = True  # 甩牌扣底: x number of cards
    # --- settlement (units of 底注) ---
    da_guang_xian: int = 3      # 闲 scored 0
    xiao_guang_xian: int = 2    # 闲 scored 5..xiao_guang_max
    xiao_guang_max: int = 30
    bao_zhuang_xian: int = 1    # 庄 kept the contract
    kua_zhuang_xian: int = 1    # 闲 reached the bid
    xiao_dao_margin: int = 40   # 闲 >= bid + 40 -> 小倒 (x2)
    da_dao_margin: int = 70     # 闲 >= bid + 70 -> 大倒 (x3)
    shuang_jin_dan_chu: bool = False  # 双进单出: banker wins count double

    def bid_values(self) -> list[int]:
        return list(range(self.max_bid, self.min_bid - 1, -self.bid_step))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Rules":
        r = cls()
        for k, v in (d or {}).items():
            if hasattr(r, k):
                setattr(r, k, type(getattr(r, k))(v))
        return r


DEFAULT_RULES = Rules()
