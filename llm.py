import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def safe_get(df, row_name):
    """
    Safely fetch a financial metric from a dataframe.
    """
    try:
        if row_name in df.index:
            value = df.loc[row_name].iloc[0]
            return f"{value:,.0f}"
    except:
        pass

    return "N/A"


def analyze_stock(info, history, income, balance, cashflow):

    latest = history.iloc[-1]

    current_price = latest["Close"]
    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    ema20 = latest["EMA20"]

    highest = history["High"].max()
    lowest = history["Low"].min()
    volume = latest["Volume"]

    # -------------------------
    # Income Statement
    # -------------------------

    revenue = safe_get(income, "Total Revenue")
    gross_profit = safe_get(income, "Gross Profit")
    operating_income = safe_get(income, "Operating Income")
    net_income = safe_get(income, "Net Income")

    # -------------------------
    # Balance Sheet
    # -------------------------

    total_assets = safe_get(balance, "Total Assets")
    total_liabilities = safe_get(balance, "Total Liabilities")
    shareholder_equity = safe_get(balance, "Stockholders Equity")
    cash = safe_get(balance, "Cash And Cash Equivalents")

    # -------------------------
    # Cash Flow
    # -------------------------

    operating_cf = safe_get(cashflow, "Operating Cash Flow")
    investing_cf = safe_get(cashflow, "Investing Cash Flow")
    financing_cf = safe_get(cashflow, "Financing Cash Flow")
    free_cf = safe_get(cashflow, "Free Cash Flow")

    prompt = f"""
You are a Senior Equity Research Analyst at JPMorgan.

Your reports are written for institutional investors.

Only use the supplied information.
If information is insufficient, explicitly mention that instead of guessing.

=================================================
COMPANY INFORMATION
=================================================

Company : {info["Company"]}

Sector : {info["Sector"]}

Industry : {info["Industry"]}

Current Price : {current_price:.2f}

Market Cap : {info["Market Cap"]}

PE Ratio : {info["PE Ratio"]}

EPS : {info["EPS"]}

Dividend Yield : {info["Dividend Yield"]}

52 Week High : {highest:.2f}

52 Week Low : {lowest:.2f}

Latest Trading Volume : {volume:,.0f}

=================================================
TECHNICAL INDICATORS
=================================================

20 Day SMA : {sma20:.2f}

50 Day SMA : {sma50:.2f}

20 Day EMA : {ema20:.2f}

=================================================
BUSINESS SUMMARY
=================================================

{info["Summary"]}

=================================================
INCOME STATEMENT
=================================================

Revenue : {revenue}

Gross Profit : {gross_profit}

Operating Income : {operating_income}

Net Income : {net_income}

=================================================
BALANCE SHEET
=================================================

Total Assets : {total_assets}

Total Liabilities : {total_liabilities}

Shareholder Equity : {shareholder_equity}

Cash & Equivalents : {cash}

=================================================
CASH FLOW
=================================================

Operating Cash Flow : {operating_cf}

Investing Cash Flow : {investing_cf}

Financing Cash Flow : {financing_cf}

Free Cash Flow : {free_cf}

=================================================

Generate a professional equity research report with the following sections:

# Executive Summary

Brief overview of the company.

# Fundamental Analysis

Discuss valuation, profitability, earnings and business quality.

# Technical Analysis

Interpret moving averages, price behaviour and momentum.

# Financial Health

Discuss assets, liabilities, cash flow and financial stability.

# Key Strengths

Provide bullet points.

# Key Risks

Provide bullet points.

# Growth Opportunities

Mention future growth drivers.

# Short-Term Outlook

Discuss momentum over the next few months.

# Long-Term Outlook

Discuss prospects over the next few years.

# Final Investment Opinion

Provide a balanced opinion with reasoning.

Avoid making a definitive "buy", "sell", or "hold" recommendation. Instead, explain the factors an investor should weigh based on the available information.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1800
    )

    return response.choices[0].message.content