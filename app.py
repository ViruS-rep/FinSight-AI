import streamlit as st
import plotly.graph_objects as go

from stock import StockAnalyzer
from llm import analyze_stock

def format_number(x):
    if x is None:
        return ""

    if isinstance(x, (int, float)):
        abs_x = abs(x)

        if abs_x >= 1_000_000_000_000:
            return f"{x/1_000_000_000_000:.2f} T"

        elif abs_x >= 1_000_000_000:
            return f"{x/1_000_000_000:.2f} B"

        elif abs_x >= 1_000_000:
            return f"{x/1_000_000:.2f} M"

        elif abs_x >= 1_000:
            return f"{x/1_000:.2f} K"

        else:
            return f"{x:.2f}"

    return x

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="FinSight AI",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("📈 FinSight AI")

st.sidebar.caption("LLM-Powered Stock Research & Financial Analysis")

st.sidebar.caption(
    "Examples: **AAPL**, **MSFT**, **NVDA**, **GOOGL**, **TSLA**, **RELIANCE.NS**, **TCS.NS**, **INFY.NS**"
)

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL",
    placeholder="e.g. AAPL or RELIANCE.NS"
)

period = st.sidebar.selectbox(
    "Time Period",
    ["1mo", "3mo", "6mo", "1y", "5y"],
    index=3
)

analyze = st.sidebar.button("🔍 Analyze")

# ---------------------------------------------------
# Main App
# ---------------------------------------------------
st.title("📈 FinSight AI")
st.caption("AI Powered Financial Research Assistant")

if analyze:

    analyzer = StockAnalyzer(ticker)
    info = analyzer.get_info()
    history = analyzer.get_history(period)

    st.header(info["Company"])

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Current Price", info["Current Price"])

    with c2:
        st.metric("Market Cap", format_number(info["Market Cap"]))

    with c3:
        st.metric("PE Ratio", info["PE Ratio"])

    with c4:
        st.metric("EPS", info["EPS"])

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.metric("Dividend Yield", info["Dividend Yield"])

    with c6:
        st.metric("52 Week High", info["52 Week High"])

    with c7:
        st.metric("52 Week Low", info["52 Week Low"])

    with c8:
        st.metric("Sector", info["Sector"])

    st.divider()

    # ---------------------------------------------------
    # Business Summary
    # ---------------------------------------------------

    st.subheader("🏢 Company Overview")
    st.write(info["Summary"])

    st.divider()

    # ---------------------------------------------------
    # Candlestick Chart
    # ---------------------------------------------------

    st.subheader("📈 Stock Price Analysis")

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=history.index,
            open=history["Open"],
            high=history["High"],
            low=history["Low"],
            close=history["Close"],
            name="Candlestick"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["SMA20"],
            name="SMA 20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["SMA50"],
            name="SMA 50"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["EMA20"],
            name="EMA 20"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        title=f"{ticker.upper()} Price Chart"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # Volume
    # ---------------------------------------------------

    st.subheader("📊 Trading Volume")

    volume_fig = go.Figure()

    volume_fig.add_trace(
        go.Bar(
            x=history.index,
            y=history["Volume"],
            name="Volume"
        )
    )

    volume_fig.update_layout(
        template="plotly_dark",
        height=300
    )

    st.plotly_chart(volume_fig, use_container_width=True)

    # ---------------------------------------------------
    # Statistics
    # ---------------------------------------------------

    st.subheader("📊 Key Statistics")

    latest = history.iloc[-1]

    s1, s2, s3 = st.columns(3)

    with s1:
        st.metric("Latest Close", f"${latest['Close']:.2f}")

    with s2:
        st.metric("Highest Price", f"${history['High'].max():.2f}")

    with s3:
        st.metric("Lowest Price", f"${history['Low'].min():.2f}")

    st.divider()

    # ---------------------------------------------------
    # Financial Statements
    # ---------------------------------------------------

    st.header("📑 Financial Statements")

    income_tab, balance_tab, cash_tab = st.tabs(
        [
            "Income Statement",
            "Balance Sheet",
            "Cash Flow"
        ]
    )

    with income_tab:
        income = analyzer.get_income_statement()

        income = income.map(format_number)
        
        st.dataframe(
            income,
            use_container_width=True,
            height=500
        )

    with balance_tab:
        balance = analyzer.get_balance_sheet()

        balance = balance.map(format_number)
        st.dataframe(
            balance,
            use_container_width=True,
            height=500
        )

    with cash_tab:
        cash = analyzer.get_cashflow()

        cash = cash.map(format_number)
        st.dataframe(
            cash,
            use_container_width=True,
            height=500
        )

    st.divider()

    # ---------------------------------------------------
    # AI Analysis
    # ---------------------------------------------------

    st.header("🤖 AI Investment Analysis")

    with st.spinner("Generating AI Report..."):
        report = analyze_stock(
            info,
            history,
            analyzer.get_income_statement(),
            analyzer.get_balance_sheet(),
            analyzer.get_cashflow()
)

    with st.container(border=True):
        st.markdown(report)
