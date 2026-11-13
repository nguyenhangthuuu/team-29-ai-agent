# src/pipeline/etf_universe.py
from dataclasses import dataclass
from typing import List

@dataclass
class ETFMeta:
    id: int
    theme: str
    ticker: str
    isin: str
    yf_symbol: str
    full_name: str
    currency: str
    leverage: float
    is_leveraged_or_short: bool

# Mapped from the official QDC slides (Slides 9 & 10)
ETF_UNIVERSE: List[ETFMeta] = [
    # Global & Regional Equities
    ETFMeta(1, "Global equity (benchmark)", "WORLD", "IE00B4L5Y983", "IWDA.AS", "iShares Core MSCI World UCITS ETF", "EUR", 1.0, False),
    ETFMeta(2, "US equity", "S&P 500", "IE00B5BMR087", "SXR8.DE", "iShares Core S&P 500 UCITS ETF", "EUR", 1.0, False),
    ETFMeta(3, "European equity", "EUROPE 600", "LU0908500753", "MEUD.PA", "Amundi Core Stoxx Europe 600 UCITS ETF", "EUR", 1.0, False),
    ETFMeta(4, "European equity", "EURO STOXX 50", "IE00B53L3W79", "CSX5.DE", "iShares Core EURO STOXX 50 UCITS ETF", "EUR", 1.0, False),
    ETFMeta(5, "European equity", "IBEX 35", "FR0010655746", "2A65.DE", "Amundi IBEX 35 UCITS ETF", "EUR", 1.0, False),
    ETFMeta(6, "Home market", "BEL 20", "FR0000021842", "BEL.PA", "Amundi BEL 20 UCITS ETF", "EUR", 1.0, False),
    
    # Factors & Thematic Tech
    ETFMeta(7, "Size & style factors", "WORLD SMALL CAP", "IE00BF4RFH31", "IUSN.DE", "iShares MSCI World Small Cap UCITS ETF", "EUR", 1.0, False),
    ETFMeta(8, "Size & style factors", "WORLD MIN VOL", "IE00B8FHGS14", "MVOL.L", "iShares Edge MSCI World Minimum Volatility", "USD", 1.0, False),
    ETFMeta(9, "Size & style factors", "DIVIDEND LEADERS", "NL0011683594", "TDIV.AS", "VanEck Developed Markets Dividend Leaders", "EUR", 1.0, False),
    ETFMeta(10, "Income strategies", "NASDAQ INCOME", "IE000U9J8HX9", "JEQP.DE", "JPMorgan Nasdaq Equity Premium Income", "EUR", 1.0, False),
    ETFMeta(11, "Equity / Tech", "NASDAQ", "IE00B53SZB19", "CSNDX.DE", "iShares NASDAQ 100 UCITS ETF", "EUR", 1.0, False),
    ETFMeta(12, "Equity / Tech", "SEMIS", "IE000I8KRLL9", "SEC0.DE", "iShares MSCI Global Semiconductors UCITS ETF", "EUR", 1.0, False),
    ETFMeta(13, "Equity / Tech", "CYBERSEC", "IE00BYPLS672", "ISPY.L", "L&G Cyber Security UCITS ETF", "USD", 1.0, False),
    ETFMeta(14, "Equity / Tech", "BIOTECH", "IE00BQ70R696", "SBIO.L", "Invesco NASDAQ Biotech UCITS ETF", "USD", 1.0, False),
    ETFMeta(15, "Equity / Tech", "ROBOTICS", "IE00BYZK4552", "RBOT.L", "iShares Automation & Robotics UCITS ETF", "USD", 1.0, False),
    ETFMeta(16, "Frontier tech", "SPACE", "IE000YU9K6K2", "JEDI.DE", "VanEck Space Innovators UCITS ETF", "EUR", 1.0, False),
    ETFMeta(17, "Frontier tech", "QUANTUM", "IE0007Y8Y157", "QTUM.DE", "Defiance Quantum Computing UCITS ETF", "EUR", 1.0, False),
    ETFMeta(18, "Financials", "EURO BANKS", "LU1829219390", "BNKE.PA", "Amundi Euro Stoxx Banks UCITS ETF", "EUR", 1.0, False),
    ETFMeta(19, "Health care", "HEALTH CARE", "IE00BM67HK77", "XDWH.DE", "Xtrackers MSCI World Health Care UCITS ETF", "EUR", 1.0, False),

    # Emerging Markets
    ETFMeta(20, "Emerging markets", "EMMKT", "IE00BKM4GZ66", "EMIM.AS", "iShares Core MSCI Emerging Markets IMI", "EUR", 1.0, False),
    ETFMeta(21, "Emerging markets", "CHINA LC", "IE00B02KXK85", "FXC.DE", "iShares China Large Cap UCITS ETF", "EUR", 1.0, False),
    ETFMeta(22, "Emerging markets", "INDIA", "IE00BZCQB185", "NDIA.DE", "iShares MSCI India UCITS ETF", "EUR", 1.0, False),
    ETFMeta(23, "Emerging markets", "KOREA", "LU1900066975", "KRW.PA", "Amundi MSCI Korea UCITS ETF", "EUR", 1.0, False),
    ETFMeta(24, "Emerging markets", "BRAZIL", "IE00B0M63516", "4BRZ.DE", "iShares MSCI Brazil UCITS ETF", "EUR", 1.0, False),
    ETFMeta(25, "Emerging markets", "CHINA A", "IE00BQT3WG13", "CNYA.L", "iShares MSCI China A UCITS ETF", "USD", 1.0, False),

    # Energy, Metals & Commodities
    ETFMeta(26, "Energy & commodities", "WORLD ENERGY", "IE00BM67HM91", "XDW0.DE", "Xtrackers MSCI World Energy UCITS ETF", "EUR", 1.0, False),
    ETFMeta(27, "Energy & commodities", "OIL & GAS E&P", "IE00B6R51Z18", "SPOG.L", "iShares Oil & Gas Exploration & Production", "USD", 1.0, False),
    ETFMeta(28, "Energy transition", "URANIUM", "IE000M7V94E1", "URNU.DE", "VanEck Uranium and Nuclear Technologies", "EUR", 1.0, False),
    ETFMeta(29, "Energy transition", "CLEAN ENERGY", "IE00BKSBCH80", "INRG.L", "iShares Global Clean Energy UCITS ETF", "USD", 1.0, False),
    ETFMeta(30, "Energy transition", "HYDROGEN", "FR0010930644", "HYDR.PA", "Amundi Global Hydrogen UCITS ETF", "EUR", 1.0, False),
    ETFMeta(31, "Industrial metals", "COPPER MINERS", "IE0003Z9E2Y3", "COPG.L", "Global X Copper Miners UCITS ETF", "USD", 1.0, False),
    ETFMeta(32, "Industrial metals", "RARE EARTHS", "IE0002PG6CA6", "REMX.DE", "VanEck Rare Earth and Strategic Metals", "EUR", 1.0, False),
    ETFMeta(33, "Precious metals", "GOLD", "IE00B4ND3602", "SGLN.L", "iShares Physical Gold ETC", "USD", 1.0, False),
    ETFMeta(34, "Precious metals", "GOLD MINERS", "IE00BQQP9F84", "GDX.DE", "VanEck Gold Miners UCITS ETF", "EUR", 1.0, False),
    ETFMeta(35, "Precious metals", "SILVER MINERS", "IE000UL6CLP7", "SILV.L", "Global X Silver Miners UCITS ETF", "USD", 1.0, False),

    # Bonds & Cash
    ETFMeta(36, "Fixed income / Bonds", "HIGHY BOND", "LU1109942653", "XHYG.DE", "Xtrackers II EUR High Yield Corporate Bond", "EUR", 1.0, False),
    ETFMeta(37, "Fixed income / Bonds", "US TREASURY 20Y", "LU1407890620", "DTLA.L", "Amundi US Treasury Bond 20Y+ UCITS ETF", "USD", 1.0, False),
    ETFMeta(38, "Fixed income / Bonds", "GLOBAL AGG BOND", "IE00BG47KH54", "VAGF.DE", "Vanguard Global Aggregate Bond EUR Hedged", "EUR", 1.0, False),
    ETFMeta(39, "Fixed income / Bonds", "EUR GOVT BOND", "IE00B4WXJJ64", "SEGA.DE", "iShares Core EUR Govt Bond UCITS ETF", "EUR", 1.0, False),
    ETFMeta(40, "Money market (cash)", "OVERNIGHT CASH", "LU1190417599", "XEON.DE", "Amundi Smart Overnight Return UCITS ETF", "EUR", 1.0, False),

    # Defensive & Digital Assets
    ETFMeta(41, "Defensive sector", "DEFENSE", "IE000YYE6WK5", "DFNS.DE", "VanEck Defense UCITS ETF", "EUR", 1.0, False),
    ETFMeta(42, "Defensive sector", "STAPLES", "IE00BM67HN09", "XDWS.DE", "Xtrackers MSCI World Consumer Staples", "EUR", 1.0, False),
    ETFMeta(43, "Digital assets", "BITCOIN", "XS2940466316", "IBIT.DE", "iShares Bitcoin ETP", "EUR", 1.0, False),
    ETFMeta(44, "Digital assets", "CRYPTO EQUITY", "IE00BMDKNW35", "DAPP.DE", "VanEck Crypto and Blockchain Innovators", "EUR", 1.0, False),

    # Leveraged & Inverse ETFs (CRITICAL FLAGS SET)
    ETFMeta(45, "Leveraged & inverse", "STOXX SHORT", "FR0010424143", "2XSB.PA", "Amundi EURO STOXX 50 Daily (-2x) Inverse", "EUR", -2.0, True),
    ETFMeta(46, "Leveraged & inverse", "COPPER LVG", "IE00B8JVMZ80", "3COP.L", "WisdomTree Copper 3x Daily Leveraged", "USD", 3.0, True),
    ETFMeta(47, "Leveraged & inverse", "PALLADIUM LVG", "IE00B94QLN63", "2PAL.L", "WisdomTree Palladium 2x Daily Leveraged", "USD", 2.0, True),
    ETFMeta(48, "Leveraged & inverse", "NASDAQ 3X", "IE00BLRPRL42", "QQQ3.L", "WisdomTree NASDAQ 100 3x Daily Leveraged", "USD", 3.0, True),
    ETFMeta(49, "Leveraged & inverse", "S&P 500 SHORT", "LU0322251520", "XUPS.DE", "Xtrackers S&P 500 Inverse Daily Swap", "EUR", -1.0, True),
    ETFMeta(50, "Leveraged & inverse", "EURO STOXX 2X", "FR0010468983", "RSX2.PA", "Amundi EURO STOXX 50 Daily (2x) Leveraged", "EUR", 2.0, True),
]