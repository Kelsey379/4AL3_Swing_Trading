import pandas as pd
import yfinance as yf

ticker_string = "AAPL SOXX SPAB SPCE SPDW SPEM SPEU SPG SPGI SPGP SPHB SPHQ SPIB SPIP SPLB SPLG SPLK SPLV SPMB SPMD SPOT SPSB SPTI SPTL SPTM SPVM SPXL SPXS SPY SPYG SPYV SQ SQSP SRE SRS SSG SSO STE STI STNE STT STX STZ SU SUSA SWAV SWBI SWCH SWI SWK SWKS SWN SYF SYK SYY SZK T TAL TAP TCOM TDC TDG TDOC TDY TEAM TECH TECL TECS TEL TER TEVA TFC TFI TFX TGT TIP TJX TLH TLRY TLT TM TME TMF TMO TMUS TMV TNA TOK TOL TOST TOTS TPR TPTX TRIP TRMB TROW TRU TRUE TRV TSCO TSLA TSM TSN TSO TSP TST TT TTC TTD TTGT TTWO TUP TV TVIX TWLO TWTR TXN TXT TYME TZOO"

print("daily backdrop data")
daily_data = yf.download(ticker_string, period="1mo", interval="1d", group_by="ticker")
daily_data.to_csv("daily_backdrop.csv")
print("hourly intraday trigger data")
hourly_raw = yf.download(ticker_string, period="1mo", interval="1h", group_by="ticker")
print("resample multi-our data to 4h blocks")

#aggregation rules
resample_rules = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Adj Close': 'last',
    'Volume': 'sum'
}

# sample tickers independently
processed_tickers = {}
tickers_list = ticker_string.split()

for ticker in tickers_list:
    if ticker in hourly_raw.columns.levels[0]:
        ticker_df = hourly_raw[ticker].dropna(how='all')
        if not ticker_df.empty:
            # resample ticker data to 4h/drop empty intervals
            resampled_df = ticker_df.resample('4h', origin='start_day').agg({
                col: resample_rules[col] for col in ticker_df.columns if col in resample_rules
            })
            resampled_df = resampled_df[resampled_df['Volume'] > 0]
            processed_tickers[ticker] = resampled_df

# concat ticker data into one timeframe
four_hour_data = pd.concat(processed_tickers, axis=1, names=['Ticker', 'Price'])
four_hour_data.to_csv("4hour_triggers.csv")
print("all data saved")
