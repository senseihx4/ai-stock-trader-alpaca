STOCKS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AMD", "INTC", "ORCL",
    "CRM", "ADBE", "NFLX", "PYPL", "SHOP", "UBER", "SNAP", "LYFT", "CMCSA", "QCOM",

    # Finance
    "JPM", "BAC", "GS", "MS", "C", "WFC", "AXP", "COF", "SCHW", "BLK",

    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "BMY", "GILD", "AMGN", "UNH", "TMO", "ABT",

    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "BP", "ONGC.NS",

    # Consumer
    "WMT", "COST", "MCD", "SBUX", "NKE", "KO", "PG", "TGT", "DIS",
]

STOCKS = list(dict.fromkeys(STOCKS))  # remove duplicates
