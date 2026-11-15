import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Adds running headers and footers with page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "QTEM Data Challenge 2026-2027 | ReBel Agentic AI | Team 29")
            self.drawRightString(558, 750, "50 ETF Universe EDA & Decision Framework")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        
        self.drawString(54, 32, "Confidential — Prepared for QDC Evaluators, Belfius & QuantumBlack")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf(filename="QDC_50_ETF_Universe_EDA_Summary.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#8A1538")    # Belfius Red/Burgundy
    SECONDARY = colors.HexColor("#1A365D")  # Deep Navy
    DARK_TEXT = colors.HexColor("#2D3748")  # Slate Charcoal
    LIGHT_BG = colors.HexColor("#F7FAFC")   # Off-white card
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=PRIMARY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=SECONDARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=DARK_TEXT,
        spaceAfter=6
    )
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=DARK_TEXT
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=DARK_TEXT
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=DARK_TEXT
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("ReBel ETF Universe: Exploratory Data Analysis & Decision Framework", title_style))
    story.append(Paragraph("QTEM Data Challenge 2026–2027 | Fall Edition | Belfius x QuantumBlack | Team 29", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceBefore=0, spaceAfter=10))

    # Executive Overview
    story.append(Paragraph("Executive Overview", h1_style))
    overview_text = (
        "This report provides the quantitative foundations of the <b>50 ETF Universe</b> specified for the "
        "<b>ReBel Agentic AI Investment Competition</b>. Using 3 full years of daily historical data "
        "(Nov 13, 2023 – Nov 13, 2026; 780 trading days) normalized to EUR via <code>EURUSD=X</code>, we establish "
        "the empirical baseline required for the multi-agent pipeline and the Human-in-the-Loop (HITL) investment committee."
    )
    story.append(Paragraph(overview_text, body_style))

    # SECTION 1: FORMULAS
    story.append(Paragraph("1. Quantitative Formulations", h1_style))
    formula_intro = (
        "To prevent look-ahead bias and handle assets with heterogeneous inception dates (e.g., Bitcoin ETP and Quantum Tech "
        "incepted after Nov 2023), all performance metrics are normalized on an asset-specific observed horizon:"
    )
    story.append(Paragraph(formula_intro, body_style))

    formula_table_data = [
        [
            Paragraph("<b>Metric</b>", table_header),
            Paragraph("<b>Mathematical Formula</b>", table_header),
            Paragraph("<b>Financial Interpretation</b>", table_header)
        ],
        [
            Paragraph("<b>CAGR (Annual Return)</b>", table_cell_bold),
            Paragraph("<i>CAGR<sub>i</sub> = (P<sub>i,last</sub> / P<sub>i,first_valid</sub>)<sup>(252 / N<sub>i</sub>)</sup> - 1</i>", table_cell),
            Paragraph("Compound annual growth rate using asset's first traded price and observed days <i>N<sub>i</sub></i>.", table_cell)
        ],
        [
            Paragraph("<b>Annualized Volatility</b>", table_cell_bold),
            Paragraph("<i>&sigma;<sub>i</sub> = &radic;252 &times; StdDev(R<sub>i,t</sub>)</i>", table_cell),
            Paragraph("Standard deviation of daily log-returns scaled by annual trading factor.", table_cell)
        ],
        [
            Paragraph("<b>Sharpe Ratio</b>", table_cell_bold),
            Paragraph("<i>Sharpe<sub>i</sub> = (CAGR<sub>i</sub> - R<sub>f</sub>) / &sigma;<sub>i</sub></i>", table_cell),
            Paragraph("Excess risk-adjusted return benchmarked against ECB risk-free rate <i>R<sub>f</sub> = 3.0%</i>.", table_cell)
        ],
        [
            Paragraph("<b>Maximum Drawdown</b>", table_cell_bold),
            Paragraph("<i>MDD<sub>i</sub> = min<sub>t</sub> [ (P<sub>i,t</sub> - max<sub>&tau;&le;t</sub> P<sub>i,&tau;</sub>) / max<sub>&tau;&le;t</sub> P<sub>i,&tau;</sub> ]</i>", table_cell),
            Paragraph("Worst peak-to-trough capital decline across the full observed timeline.", table_cell)
        ],
        [
            Paragraph("<b>Compounding Volatility Drag</b>", table_cell_bold),
            Paragraph("<i>&Delta;<sub>decay</sub> &approx; -0.5 &times; L(L - 1) &sigma;<sup>2</sup> &Delta;t</i>", table_cell),
            Paragraph("Beta slippage on daily leveraged products (<i>L &ne; 1</i>); path dependency erosion in choppy markets.", table_cell)
        ]
    ]
    f_table = Table(formula_table_data, colWidths=[1.4*inch, 2.7*inch, 2.9*inch])
    f_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(f_table)
    story.append(Spacer(1, 8))

    # SECTION 2: TOP & BOTTOM RESULTS
    story.append(Paragraph("2. Empirical Findings: Leaders vs. The Danger Zone", h1_style))
    
    story.append(Paragraph("A. Highest Risk-Adjusted Performers (Top 5 Sharpe Ratios)", h2_style))
    top5_data = [
        [Paragraph("<b>Ticker</b>", table_header), Paragraph("<b>Full Name</b>", table_header), Paragraph("<b>Theme</b>", table_header), Paragraph("<b>Ann. Return</b>", table_header), Paragraph("<b>Ann. Vol</b>", table_header), Paragraph("<b>Max DD</b>", table_header), Paragraph("<b>Sharpe</b>", table_header)],
        [Paragraph("<b>IBEX 35</b>", table_cell_bold), Paragraph("Amundi IBEX 35 UCITS", table_cell), Paragraph("European Equity", table_cell), Paragraph("+30.0%", table_cell), Paragraph("16.3%", table_cell), Paragraph("-12.6%", table_cell), Paragraph("<b>1.66</b>", table_cell_bold)],
        [Paragraph("<b>DIVIDEND LEADERS</b>", table_cell_bold), Paragraph("VanEck Morningstar Div Leaders", table_cell), Paragraph("Factor Equity", table_cell), Paragraph("+18.6%", table_cell), Paragraph("11.4%", table_cell), Paragraph("-15.3%", table_cell), Paragraph("<b>1.36</b>", table_cell_bold)],
        [Paragraph("<b>BEL 20</b>", table_cell_bold), Paragraph("Amundi BEL 20 (Home Market)", table_cell), Paragraph("Belgian Equity", table_cell), Paragraph("+20.7%", table_cell), Paragraph("13.6%", table_cell), Paragraph("-14.0%", table_cell), Paragraph("<b>1.30</b>", table_cell_bold)],
        [Paragraph("<b>HYDROGEN</b>", table_cell_bold), Paragraph("Amundi Global Hydrogen", table_cell), Paragraph("Energy Transition", table_cell), Paragraph("+25.2%", table_cell), Paragraph("17.1%", table_cell), Paragraph("-15.8%", table_cell), Paragraph("<b>1.30</b>", table_cell_bold)],
        [Paragraph("<b>WORLD</b>", table_cell_bold), Paragraph("iShares Core MSCI World", table_cell), Paragraph("Global Benchmark", table_cell), Paragraph("+18.3%", table_cell), Paragraph("13.0%", table_cell), Paragraph("-21.6%", table_cell), Paragraph("<b>1.18</b>", table_cell_bold)]
    ]
    t5_table = Table(top5_data, colWidths=[1.1*inch, 1.9*inch, 1.2*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch])
    t5_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (3,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(t5_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("B. The Leveraged & Inverse Instruments (Extreme Volatility Drag)", h2_style))
    lev_data = [
        [Paragraph("<b>Ticker</b>", table_header), Paragraph("<b>Leverage Factor</b>", table_header), Paragraph("<b>Ann. Return</b>", table_header), Paragraph("<b>Ann. Vol</b>", table_header), Paragraph("<b>Max Drawdown</b>", table_header), Paragraph("<b>Sharpe</b>", table_header), Paragraph("<b>Risk Profile</b>", table_header)],
        [Paragraph("<b>PALLADIUM LVG</b>", table_cell_bold), Paragraph("2.0x Daily", table_cell), Paragraph("<b>-24.6%</b>", table_cell), Paragraph("88.1%", table_cell), Paragraph("<b>-85.3%</b>", table_cell), Paragraph("-0.31", table_cell), Paragraph("Severe commodity erosion", table_cell)],
        [Paragraph("<b>COPPER LVG</b>", table_cell_bold), Paragraph("3.0x Daily", table_cell), Paragraph("+5.9%", table_cell), Paragraph("80.1%", table_cell), Paragraph("<b>-72.3%</b>", table_cell), Paragraph("+0.04", table_cell), Paragraph("Extreme volatility drag", table_cell)],
        [Paragraph("<b>STOXX SHORT</b>", table_cell_bold), Paragraph("-2.0x Daily", table_cell), Paragraph("<b>-26.5%</b>", table_cell), Paragraph("30.3%", table_cell), Paragraph("<b>-63.3%</b>", table_cell), Paragraph("-0.97", table_cell), Paragraph("Structural inverse decay", table_cell)],
        [Paragraph("<b>S&P 500 SHORT</b>", table_cell_bold), Paragraph("-1.0x Daily", table_cell), Paragraph("<b>-11.8%</b>", table_cell), Paragraph("15.4%", table_cell), Paragraph("<b>-38.3%</b>", table_cell), Paragraph("-0.96", table_cell), Paragraph("Constant equity headwind", table_cell)],
        [Paragraph("<b>NASDAQ 3X</b>", table_cell_bold), Paragraph("3.0x Daily", table_cell), Paragraph("+36.3%", table_cell), Paragraph("56.1%", table_cell), Paragraph("<b>-60.0%</b>", table_cell), Paragraph("+0.59", table_cell), Paragraph("High return, -60% crash risk", table_cell)],
        [Paragraph("<b>EURO STOXX 2X</b>", table_cell_bold), Paragraph("2.0x Daily", table_cell), Paragraph("+29.4%", table_cell), Paragraph("30.5%", table_cell), Paragraph("-31.3%", table_cell), Paragraph("+0.87", table_cell), Paragraph("Moderate rally multiplier", table_cell)]
    ]
    lev_table = Table(lev_data, colWidths=[1.1*inch, 0.9*inch, 0.8*inch, 0.7*inch, 0.9*inch, 0.6*inch, 2.0*inch])
    lev_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#C53030")), # Red warning
        ('ALIGN', (1,0), (-2,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(lev_table)
    story.append(Spacer(1, 10))

    # SECTION 3: THE 4 DECISION BUCKETS
    story.append(Paragraph("3. The 4 Decision Buckets (Portfolio Architecture)", h1_style))
    bucket_intro = (
        "To enable disciplined trade execution across the 5 competition dates (Dec 7, 10, 14, 17, 21), "
        "the universe is structured into 4 functional decision buckets:"
    )
    story.append(Paragraph(bucket_intro, body_style))

    bucket_summary_data = [
        [
            Paragraph("<b>Decision Bucket</b>", table_header),
            Paragraph("<b>Count</b>", table_header),
            Paragraph("<b>Avg Return</b>", table_header),
            Paragraph("<b>Avg Vol</b>", table_header),
            Paragraph("<b>Avg MaxDD</b>", table_header),
            Paragraph("<b>Avg Sharpe</b>", table_header),
            Paragraph("<b>Mandatory Allocation Rule</b>", table_header)
        ],
        [
            Paragraph("<b>Bucket 1: Core Compounding Anchors</b>", table_cell_bold),
            Paragraph("21", table_cell),
            Paragraph("+13.6%", table_cell),
            Paragraph("23.5%", table_cell),
            Paragraph("-27.2%", table_cell),
            Paragraph("<b>+0.71</b>", table_cell_bold),
            Paragraph("Foundation: 40% – 60% of total capital.<br/>Max 20% single position cap.", table_cell)
        ],
        [
            Paragraph("<b>Bucket 2: Tactical Momentum Drivers</b>", table_cell_bold),
            Paragraph("15", table_cell),
            Paragraph("<b>+20.0%</b>", table_cell_bold),
            Paragraph("28.1%", table_cell),
            Paragraph("-32.6%", table_cell),
            Paragraph("+0.59", table_cell),
            Paragraph("Alpha generation: 20% – 35% of capital.<br/>Deploy on confirmed Golden Trends.", table_cell)
        ],
        [
            Paragraph("<b>Bucket 3: Capital Preservation & Buffers</b>", table_cell_bold),
            Paragraph("6", table_cell),
            Paragraph("+4.3%", table_cell),
            Paragraph("<b>7.9%</b>", table_cell),
            Paragraph("<b>-10.3%</b>", table_cell),
            Paragraph("-0.35", table_cell),
            Paragraph("Liquidity buffer: 5% – 25% of capital.<br/>Overnight Cash + Treasuries + Gold.", table_cell)
        ],
        [
            Paragraph("<b>Bucket 4: High-Beta Tactical & Hedging</b>", table_cell_bold),
            Paragraph("8", table_cell),
            Paragraph("+3.2%", table_cell),
            Paragraph("<b>50.8%</b>", table_cell_bold),
            Paragraph("<b>-57.9%</b>", table_cell_bold),
            Paragraph("-0.10", table_cell),
            Paragraph("Tactical only: 0% – 15% combined.<br/>Max 8% single position, &lt;5 days horizon.", table_cell)
        ]
    ]
    b_table = Table(bucket_summary_data, colWidths=[1.8*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.8*inch, 0.7*inch, 1.8*inch])
    b_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('ALIGN', (1,0), (-2,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(b_table)
    story.append(Spacer(1, 10))

    # SECTION 4: HUMAN IN THE LOOP CHEAT SHEET
    story.append(Paragraph("4. Human-in-the-Loop Trading Decision Matrix", h1_style))
    matrix_data = [
        [
            Paragraph("<b>Macro Regime</b>", table_header),
            Paragraph("<b>Technical Agent Signal</b>", table_header),
            Paragraph("<b>Target Bucket Allocation</b>", table_header),
            Paragraph("<b>Human Sign-Off Guideline</b>", table_header)
        ],
        [
            Paragraph("<b>Risk-On / Growth</b>", table_cell_bold),
            Paragraph("Bullish (Score &gt; +0.40)", table_cell),
            Paragraph("50% Bucket 1 + 35% Bucket 2<br/>+ 10% Cash + 5% Bucket 4", table_cell),
            Paragraph("Overweight <b>SEMIS, CYBERSEC, DEFENSE, S&P 500</b>. Allow tactical 5% in <b>NASDAQ 3X</b>.", table_cell)
        ],
        [
            Paragraph("<b>Choppy / Sideways</b>", table_cell_bold),
            Paragraph("Neutral (-0.20 to +0.20)", table_cell),
            Paragraph("60% Bucket 1 + 25% Bucket 3<br/>+ 15% Cash Buffer", table_cell),
            Paragraph("Rotate into <b>DIVIDEND LEADERS, STAPLES, HEALTH CARE, OVERNIGHT CASH</b>. Zero leveraged exposure.", table_cell)
        ],
        [
            Paragraph("<b>Risk-Off / Shock</b>", table_cell_bold),
            Paragraph("Bearish (Score &lt; -0.30)", table_cell),
            Paragraph("30% Bucket 1 + 35% Bucket 3<br/>+ 25% Cash + 10% Short", table_cell),
            Paragraph("Overweight <b>OVERNIGHT CASH, GOLD, US TREASURY 20Y</b>. Deploy hedge via <b>STOXX SHORT (-2x)</b>.", table_cell)
        ]
    ]
    m_table = Table(matrix_data, colWidths=[1.3*inch, 1.4*inch, 1.8*inch, 2.5*inch])
    m_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(m_table)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ Generated executive PDF: {filename}")

if __name__ == "__main__":
    out_dir = Path("deliverables/06_appendices")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "QDC_50_ETF_Universe_EDA_Summary.pdf"
    build_pdf(str(pdf_path))
