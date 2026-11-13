# src/pipeline/etf_universe.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ETFMeta:
    ticker_internal: str
    yf_symbol: str
    name: str
    currency: str
    theme: str
    leverage: float  # 1.0 = normal, 2.0 = 2x, -1.0 = 1x short, -2.0 = -2x short, 3.0 = 3x
    is_leveraged_or_short: bool

ETF_UNIVERSE = [
    # Benchmarks & Equities
    ETFMeta("WORLD", "IWDA.AS", "iShares Core MSCI World UCITS ETF", "EUR", "Global Equity", 1.0, False),
    ETFMeta("S&P 500", "SXR8.DE", "iShares Core S&P 500 UCITS ETF", "EUR", "US Equity", 1.0, False),
    ETFMeta("BEL 20", "BEL.PA", "Amundi BEL 20 UCITS ETF", "EUR", "Home Market", 1.0, False),
    ETFMeta("NASDAQ", "CNDX.L", "iShares NASDAQ 100 UCITS ETF", "USD", "Tech", 1.0, False),
    
    # Leveraged & Inverse (Flags set to True)
    ETFMeta("NASDAQ 3X", "QQQ3.L", "WisdomTree NASDAQ 100 3x Daily Leveraged", "USD", "Leveraged", 3.0, True),
    ETFMeta("STOXX SHORT", "2XSB.PA", "Amundi EURO STOXX 50 Daily (-2x) Inverse", "EUR", "Inverse", -2.0, True),
    ETFMeta("S&P 500 SHORT", "XUPS.DE", "Xtrackers S&P 500 Inverse Daily Swap", "EUR", "Inverse", -1.0, True),
    ETFMeta("COPPER LVG", "3COP.L", "WisdomTree Copper 3x Daily Leveraged", "USD", "Leveraged", 3.0, True),
    ETFMeta("EURO STOXX 2X", "RSX2.PA", "Amundi EURO STOXX 50 Daily (2x) Leveraged", "EUR", "Leveraged", 2.0, True),
    # ... expand to complete the 50 ETFs
]