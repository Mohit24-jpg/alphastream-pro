import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import time
import feedparser
from openai import OpenAI
import yfinance as yf

# 1. PAGE CONFIG
st.set_page_config(layout="wide", page_title="AlphaStream Pro", page_icon="📈")

# --- 2. ROBUST SESSION HANDLER ---
def create_snowpark_session():
    if "connections" in st.secrets and "snowflake" in st.secrets["connections"]:
        return Session.builder.configs(st.secrets["connections"]["snowflake"]).create()
    else:
        return get_active_session()

try:
    session = create_snowpark_session()
except Exception as e:
    st.error(f"Could not connect to Snowflake: {e}")
    st.stop()

# --- 3. THE ENGINE ---
def update_market_data():
    if "openai" in st.secrets:
        with st.spinner('AlphaStream Agent is syncing live market intelligence...'):
            try:
                # A. Fetch News
                RSS_FEEDS = [
                    "https://finance.yahoo.com/news/rssindex",
                    "http://feeds.marketwatch.com/marketwatch/topstories/",
                    "https://feeds.bloomberg.com/markets/news.rss"
                ]
                articles = []
                for feed in RSS_FEEDS:
                    try:
                        d = feedparser.parse(feed)
                        for entry in d.entries[:3]: 
                            articles.append({"TITLE": entry.title, "URL": entry.link})
                    except: pass
                
                # B. Score with AI
                client = OpenAI(api_key=st.secrets["openai"]["api_key"])
                new_tickers = []
                
                for art in articles:
                    try:
                        prompt = f"Analyze: '{art['TITLE']}'. Output: TICKER|EVENT|SCORE (-1.0 to 1.0). If no specific ticker, use MARKET."
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0
                        )
                        raw = response.choices[0].message.content.strip().split('|')
                        if len(raw) == 3 and raw[0] != 'MARKET':
                            ticker, event, score = raw[0].strip(), raw[1].strip(), float(raw[2])
                            if 'TSMC' in ticker: ticker = 'TSM' 
                            
                            # Insert News
                            session.sql(f"""
                                INSERT INTO FINANCE_DB.RAW_DATA.NEWS_SENTIMENT (PUBLISH_DATE, TITLE, URL, TICKER, EVENT_TYPE, SENTIMENT_SCORE)
                                SELECT CURRENT_TIMESTAMP(), '{art['TITLE'].replace("'", "''")}', '{art['URL']}', '{ticker}', '{event}', {score}
                                WHERE NOT EXISTS (SELECT 1 FROM FINANCE_DB.RAW_DATA.NEWS_SENTIMENT WHERE TITLE = '{art['TITLE'].replace("'", "''")}')
                            """).collect()
                            new_tickers.append(ticker)
                    except: pass

                # C. Update Prices
                if new_tickers:
                    unique_tickers = list(set(new_tickers))
                    for symbol in unique_tickers:
                        try:
                            if symbol in ['GATHER AI', 'UNKNOWN']: continue 
                            stock = yf.Ticker(symbol)
                            hist = stock.history(period="1d")
                            if not hist.empty:
                                curr = hist['Close'].iloc[-1]
                                open_p = hist['Open'].iloc[-1]
                                chg = ((curr - open_p)/open_p)*100
                                info = stock.info
                                pe = info.get('forwardPE', 0)
                                pb = info.get('priceToBook', 0)
                                rate = info.get('recommendationKey', 'none')
                                
                                session.sql(f"""
                                    MERGE INTO FINANCE_DB.RAW_DATA.MARKET_PRICES AS target
                                    USING (SELECT '{symbol}' AS T, {curr} AS C, {chg} AS P, {pe} AS PE, {pb} AS PB, '{rate}' AS R) AS source
                                    ON target.TICKER = source.T
                                    WHEN MATCHED THEN UPDATE SET CURRENT_PRICE = source.C, CHANGE_PERCENT = source.P, PE_RATIO = source.PE, PRICE_TO_BOOK = source.PB, ANALYST_RATING = source.R
                                    WHEN NOT MATCHED THEN INSERT (TICKER, CURRENT_PRICE, CHANGE_PERCENT, PE_RATIO, PRICE_TO_BOOK, ANALYST_RATING) VALUES (source.T, source.C, source.P, source.PE, source.PB, source.R)
                                """).collect()
                        except: pass
                st.toast("System Sync Complete")
            except Exception as e:
                st.warning(f"Sync issue (minor): {e}")

# --- PROFESSIONAL CSS & MOBILE FIX ---
st.markdown("""
<style>
    /* 1. FORCE LIGHT MODE & GLOBAL FONT FIX */
    :root { color-scheme: light only; }
    .stApp { background-color: #f5f7f9; font-family: 'Inter', sans-serif; color: #172b4d; }
    h1 { color: #172b4d; font-weight: 800; margin-bottom: 0px; }
    h2, h3, p, div, label, span { color: #172b4d !important; } 
    
    /* 2. TAB HIGHLIGHTING RESTORED (THE "BLUE TABS") */
    button[data-baseweb="tab"] {
        background-color: white !important;
        border: 1px solid #e1e4e8 !important;
        color: #5e6c84 !important;
        border-radius: 4px;
        margin-right: 8px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #e3f2fd !important;
        border: 1px solid #0052cc !important;
        color: #0052cc !important;
        font-weight: bold;
        border-bottom: 3px solid #0052cc !important;
    }

    /* 3. DROPDOWN VISIBILITY FIX */
    .stMultiSelect { z-index: 999; }
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important;
        border-color: #e1e4e8 !important;
    }

    /* 4. HIDE CLUTTER */
    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 5. DESKTOP CARDS & CHART CENTERING */
    div.css-1r6slb0, div.stDataFrame, div.stPlotlyChart {
        background-color: white; padding: 24px; border-radius: 8px; border: 1px solid #e1e4e8; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        display: flex; justify-content: center; /* Center charts */
        align-items: center;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff; padding: 16px; border-radius: 8px; border: 1px solid #e1e4e8; border-left: 4px solid #0052cc;
    }
    
    /* 6. GUIDE CARDS */
    .guide-card {
        background-color: #ebf3fc; padding: 12px 18px; border-radius: 6px; 
        border-left: 4px solid #0052cc; margin-bottom: 20px; color: #172b4d; font-size: 0.85rem; line-height: 1.4;
    }
    .source-badge {
        background-color: #e3f2fd; color: #0d47a1; padding: 4px 8px; border-radius: 4px; 
        font-size: 0.8rem; font-weight: 600; display: inline-block; margin-bottom: 10px;
    }

    /* === AGGRESSIVE MOBILE OPTIMIZATION (Screens < 768px) === */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important;
        }
        [data-testid="column"] {
            width: 100% !important; min-width: 100% !important; margin-bottom: 10px !important; flex: 1 1 auto !important;
        }
        h1 { font-size: 1.6rem !important; }
        .stButton button { width: 100%; }
        
        /* Fix Chart Sizing on Mobile */
        .js-plotly-plot { width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)

# 2. LOAD DATA
query = """
SELECT 
    n.PUBLISH_DATE, n.TITLE, n.URL, n.TICKER, n.SENTIMENT_SCORE, n.EVENT_TYPE,
    m.CURRENT_PRICE, m.CHANGE_PERCENT, m.UPDATE_TIME, 
    m.PE_RATIO, m.PRICE_TO_BOOK, m.ANALYST_RATING,
    CURRENT_TIMESTAMP() as DB_NOW
FROM FINANCE_DB.RAW_DATA.NEWS_SENTIMENT n
LEFT JOIN FINANCE_DB.RAW_DATA.MARKET_PRICES m ON n.TICKER = m.TICKER
ORDER BY n.PUBLISH_DATE DESC
"""
try:
    df = session.sql(query).to_pandas()
except:
    df = pd.DataFrame()

if not df.empty:
    df = df.drop_duplicates(subset=['TITLE', 'TICKER'])
    for col in ['PE_RATIO', 'PRICE_TO_BOOK']:
        df[col] = df[col].replace(0, np.nan)
    df['ANALYST_RATING'] = df['ANALYST_RATING'].fillna("N/A")
    
    db_now = pd.to_datetime(df['DB_NOW'].iloc[0]).tz_localize('UTC') if df['DB_NOW'].iloc[0].tzinfo is None else pd.to_datetime(df['DB_NOW'].iloc[0])
    est_time = db_now.tz_convert('US/Eastern')
    last_update = est_time.strftime("%b %d, %I:%M %p EST")
else:
    last_update = "Waiting for Data..."

# --- HEADER & EXECUTIVE SUMMARY ---
c1, c2 = st.columns([6, 2])
with c1:
    st.title("AlphaStream Pro")
    st.markdown("<div style='margin-top: -5px; margin-bottom: 10px; font-size: 1.0rem; color: #5e6c84;'>Built by <b>Mohit Vaid</b></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e1e4e8; font-size: 0.9rem; margin-bottom: 15px;">
        <b>🚀 Executive Summary:</b> This dashboard uses Artificial Intelligence to read thousands of news articles in real-time and compare them against Wall Street data.
        <ul style="margin-bottom: 0;">
            <li><b>Net Sentiment Score:</b> The aggregate "Mood" of the market (-1.0 Panic to +1.0 Euphoria).</li>
            <li><b>Divergence:</b> When AI (News) disagrees with Analysts (Banks). This indicates a potential breakout.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
with c2:
    if st.button("Refresh Data", type="primary"): 
        update_market_data()
        st.rerun()
    st.markdown(f"<div style='text-align: right; color: #5e6c84; font-size: 0.75rem; margin-top: 5px;'>Last Sync: <b>{last_update}</b></div>", unsafe_allow_html=True)

st.divider()

# --- GLOBAL FILTER ---
col_search, col_space = st.columns([2, 2])
with col_search:
    all_tickers = sorted(df['TICKER'].unique().tolist()) if not df.empty else []
    selected_tickers = st.multiselect("Filter by Ticker", all_tickers)

display_df = df[df['TICKER'].isin(selected_tickers)] if selected_tickers else df

# --- KPI CARDS (SMART LOGIC) ---
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Live Articles", len(display_df))
with k2: 
    sent = display_df['SENTIMENT_SCORE'].mean() if not display_df.empty else 0
    st.metric("Net Sentiment", f"{sent:.2f}", delta=f"{sent:.2f}")
with k3:
    if len(selected_tickers) == 1:
        single_ticker = selected_tickers[0]
        ticker_data = df[df['TICKER'] == single_ticker].iloc[0]
        price = ticker_data['CURRENT_PRICE'] if pd.notna(ticker_data['CURRENT_PRICE']) else 0
        change = ticker_data['CHANGE_PERCENT'] if pd.notna(ticker_data['CHANGE_PERCENT']) else 0
        st.metric(f"{single_ticker} Price (Intraday)", f"${price:.2f}", f"{change:.2f}%")
    elif not selected_tickers:
        top = df.loc[df['CHANGE_PERCENT'].idxmax()] if not df.empty and 'CHANGE_PERCENT' in df.columns else None
        if top is not None:
            st.metric("Top Mover (Intraday)", f"{top['TICKER']} (${top['CURRENT_PRICE']:.2f})", f"{top['CHANGE_PERCENT']:.2f}%")
        else: st.metric("Top Mover", "-", "0%")
    else:
        avg_pe = display_df['PE_RATIO'].mean() if not display_df.empty else 0
        st.metric("Avg P/E Ratio", f"{avg_pe:.1f}x" if not pd.isna(avg_pe) else "N/A")

with k4: st.metric("Pipeline Status", "Active", delta="Live", delta_color="off")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Market Pulse", "Alpha Hunter", "Credibility Check"])

# === TAB 1: MARKET PULSE ===
with tab1:
    st.markdown("""
    <div class="guide-card">
        <div class="guide-title">Market Pulse: Sentiment & Themes</div>
        • <b>Mood Index:</b> Visualizes the aggregate emotion of the market. 
        <br>(<span style='color:#4CAF50'><b>Green</b></span> = Bullish, <span style='color:#E0E0E0'><b>Grey</b></span> = Neutral, <span style='color:#FF5252'><b>Red</b></span> = Bearish).
        <br>• <b>Why these tickers?</b> This dashboard is <b>Event-Driven</b>. We only display assets currently appearing in global news feeds.
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Sentiment Distribution")
        if not display_df.empty:
            display_df['Score Range'] = display_df['SENTIMENT_SCORE'].apply(lambda x: round(x * 5) / 5)
            hist_data = display_df.groupby('Score Range').size().reset_index(name='Volume')
            fig_hist = px.bar(hist_data, x="Score Range", y="Volume", color="Score Range", 
                color_continuous_scale=["#FF5252", "#E0E0E0", "#4CAF50"], range_color=[-1, 1], template="plotly_white")
            fig_hist.update_layout(height=350, bargap=0.1, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)
            
            with st.expander("Drill Down: Filter by Sentiment Range"):
                min_s, max_s = float(display_df['SENTIMENT_SCORE'].min()), float(display_df['SENTIMENT_SCORE'].max())
                if min_s == max_s: min_s -= 0.01; max_s += 0.01
                values = st.slider("Select Sentiment Range", min_s, max_s, (min_s, max_s))
                subset = display_df[(display_df['SENTIMENT_SCORE'] >= values[0]) & (display_df['SENTIMENT_SCORE'] <= values[1])]
                st.dataframe(subset[['TICKER', 'SENTIMENT_SCORE', 'TITLE', 'URL']], hide_index=True, use_container_width=True,
                    column_config={"URL": st.column_config.LinkColumn("Source", display_text="Read"), "SENTIMENT_SCORE": st.column_config.ProgressColumn("Score", min_value=-1, max_value=1, format="%.2f")})

    with c_right:
        st.subheader("Dominant Narratives")
        if not display_df.empty:
            theme_counts = display_df['EVENT_TYPE'].value_counts().reset_index()
            theme_counts.columns = ['Theme', 'Count']
            fig_donut = px.pie(theme_counts.head(7), values='Count', names='Theme', hole=0.6, color_discrete_sequence=px.colors.qualitative.G10)
            fig_donut.update_layout(height=350, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_donut, use_container_width=True)
            
            with st.expander("Drill Down: Inspect Themes & Definitions"):
                st.markdown("**Narrative Bucket Definitions:**<br>* **Macro:** Fed policy, Rates, Inflation.<br>* **Earnings:** Quarterly reports, Revenue.<br>* **Mergers:** M&A activity, Buyouts.", unsafe_allow_html=True)
                options = theme_counts['Theme'].tolist()
                theme = st.selectbox("Select Theme to Filter News:", options=options) if options else None
                if theme:
                    subset = display_df[display_df['EVENT_TYPE'] == theme][['TICKER', 'TITLE', 'URL']]
                    st.dataframe(subset, hide_index=True, use_container_width=True, column_config={"URL": st.column_config.LinkColumn("Source", display_text="Read Article")})

    # --- FORENSIC ANALYSIS SECTION ---
    st.subheader("Forensic Sentiment Analysis")
    st.markdown("Select specific articles to calculate their individual **Weight** vs. the Market Average.")
    
    if not display_df.empty:
        display_df['Impact Factor'] = display_df['SENTIMENT_SCORE'].abs()
        drivers = sorted(display_df['TITLE'].unique().tolist())
        selected_drivers = st.multiselect("Isolate Specific Headlines / Drivers:", drivers)
        global_avg = display_df['SENTIMENT_SCORE'].mean()
        
        if selected_drivers:
            subset_df = display_df[display_df['TITLE'].isin(selected_drivers)]
            subset_avg = subset_df['SENTIMENT_SCORE'].mean()
            delta = subset_avg - global_avg
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Global Market Score", f"{global_avg:.2f}")
            with m2: st.metric("Selected Driver Score", f"{subset_avg:.2f}")
            with m3: st.metric("Net Impact (Divergence)", f"{delta:.2f}", delta=f"{delta:.2f}")
            st.dataframe(subset_df[['TICKER', 'SENTIMENT_SCORE', 'TITLE', 'URL']], use_container_width=True, hide_index=True)
        else:
            st.info("👆 Select articles above to see their specific contribution to the score.")
            st.dataframe(display_df.sort_values('Impact Factor', ascending=False).head(10)[['TICKER', 'SENTIMENT_SCORE', 'TITLE', 'URL']], use_container_width=True, hide_index=True,
                column_config={"SENTIMENT_SCORE": st.column_config.ProgressColumn("Sentiment Strength", min_value=-1, max_value=1, format="%.2f"), "URL": st.column_config.LinkColumn("Evidence", display_text="Read Source"), "TITLE": "Top Headlines"})

# === TAB 2: ALPHA HUNTER ===
with tab2:
    st.markdown("""
    <div class="guide-card">
        <div class="guide-title">Alpha Hunter: Arbitrage Identification</div>
        • <b>Strategy:</b> Identify arbitrage by correlating Fundamental News (X-Axis) with Technical Price (Y-Axis).<br>
        • <b>Opportunity Zone:</b> Assets with High Sentiment (>0.5) but Lagging Price (<1%). These represent potential value disconnects.
    </div>
    """, unsafe_allow_html=True)

    c_title, c_check = st.columns([4, 1])
    with c_title: st.subheader("Price vs. Sentiment Correlation")
    with c_check: zoom_in = st.checkbox("Filter: Opportunity Zone")

    if not display_df.empty:
        scatter_df = display_df.groupby('TICKER')[['SENTIMENT_SCORE', 'CHANGE_PERCENT', 'PE_RATIO', 'ANALYST_RATING']].agg(
            {'SENTIMENT_SCORE': 'mean', 'CHANGE_PERCENT': 'mean', 'PE_RATIO': 'max', 'ANALYST_RATING': 'first'}
        ).reset_index()
        scatter_df['Color'] = scatter_df['TICKER'].apply(lambda x: '#0052cc' if selected_tickers and x in selected_tickers else ('#00873c' if scatter_df.loc[scatter_df['TICKER']==x, 'CHANGE_PERCENT'].iloc[0] > 0 else '#de350b'))
        scatter_df['ShowLabel'] = scatter_df['CHANGE_PERCENT'].abs() > scatter_df['CHANGE_PERCENT'].abs().quantile(0.8)
        scatter_df['Label'] = scatter_df.apply(lambda row: row['TICKER'] if row['ShowLabel'] else '', axis=1)
        plot_df = scatter_df[(scatter_df['SENTIMENT_SCORE'] > 0.2) & (scatter_df['CHANGE_PERCENT'] < 2)] if zoom_in else scatter_df

        fig_scatter = px.scatter(plot_df, x="SENTIMENT_SCORE", y="CHANGE_PERCENT", text="Label", color="Color", color_discrete_map="identity", hover_data=["TICKER", "PE_RATIO", "ANALYST_RATING"], template="plotly_white", size_max=60)
        fig_scatter.add_hline(y=0, line_dash="solid", line_color="#e1e4e8", line_width=1)
        fig_scatter.add_vline(x=0, line_dash="solid", line_color="#e1e4e8", line_width=1)
        fig_scatter.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='White')))
        fig_scatter.update_layout(height=500, xaxis_title="AlphaStream Sentiment Score", yaxis_title="Intraday Price Change (%)", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.dataframe(plot_df[['TICKER', 'CHANGE_PERCENT', 'SENTIMENT_SCORE', 'PE_RATIO']].sort_values(by='SENTIMENT_SCORE', ascending=False), use_container_width=True, hide_index=True,
            column_config={"CHANGE_PERCENT": st.column_config.NumberColumn("Price Change", format="%.2f %%"), "SENTIMENT_SCORE": st.column_config.ProgressColumn("Sentiment", format="%.2f", min_value=-1, max_value=1), "PE_RATIO": st.column_config.NumberColumn("P/E Ratio", format="%.1fx")})

# === TAB 3: CREDIBILITY CHECK ===
with tab3:
    st.markdown("""
    <div class="guide-card">
        <div class="guide-title">The Divergence Engine: AI vs. Wall St.</div>
        • <b>The Opinion Gap:</b> Visualizes the difference between Real-Time News (AI) and Historical Models (Analysts).<br>
        • <b>How to Read:</b> Large gaps indicate high-volatility events where the AI may be detecting news before analysts have updated their ratings.
    </div>
    """, unsafe_allow_html=True)
    
    if not display_df.empty:
        comp_df = display_df.groupby('TICKER')[['SENTIMENT_SCORE', 'ANALYST_RATING', 'PE_RATIO', 'PRICE_TO_BOOK', 'URL', 'TITLE']].agg(
            {'SENTIMENT_SCORE': 'mean', 'ANALYST_RATING': 'first', 'PE_RATIO': 'max', 'PRICE_TO_BOOK': 'max', 'URL': 'first', 'TITLE': 'first'}
        ).reset_index()

        def rating_to_score(rating):
            r = str(rating).lower()
            if 'strong buy' in r: return 1.0
            if 'buy' in r: return 0.5
            if 'sell' in r: return -0.5
            if 'underperform' in r: return -0.8
            return 0.0

        comp_df['Wall_St_Score'] = comp_df['ANALYST_RATING'].apply(rating_to_score)
        comp_df['Divergence'] = (comp_df['SENTIMENT_SCORE'] - comp_df['Wall_St_Score']).abs()
        comp_df = comp_df.sort_values('Divergence', ascending=False)

        chart_df = comp_df.head(15)
        melted_df = chart_df.melt(id_vars=['TICKER'], value_vars=['SENTIMENT_SCORE', 'Wall_St_Score'], var_name='Source', value_name='Score')
        melted_df['Source'] = melted_df['Source'].replace({'SENTIMENT_SCORE': 'AI Sentiment', 'Wall_St_Score': 'Analyst Consensus'})

        fig_side = px.bar(melted_df, x="TICKER", y="Score", color="Source", barmode='group', color_discrete_map={'AI Sentiment': '#0052cc', 'Analyst Consensus': '#97a0af'}, title="Top 15 Sentiment Divergences (Ranked by Gap)", template="plotly_white", range_y=[-1, 1])
        fig_side.update_layout(legend_title_text='', margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_side, use_container_width=True)

        st.markdown("### Fundamental Inspection (All Tickers)")
        st.dataframe(comp_df[['TICKER', 'SENTIMENT_SCORE', 'ANALYST_RATING', 'PE_RATIO', 'PRICE_TO_BOOK', 'URL']], use_container_width=True, hide_index=True,
            column_config={"SENTIMENT_SCORE": st.column_config.ProgressColumn("AlphaStream Sentiment", format="%.2f", min_value=-1, max_value=1), "ANALYST_RATING": st.column_config.TextColumn("Analyst Rating"), "PE_RATIO": st.column_config.NumberColumn("P/E (Valuation)", format="%.1fx"), "PRICE_TO_BOOK": st.column_config.NumberColumn("P/B (Assets)", format="%.1fx"), "URL": st.column_config.LinkColumn("Source", display_text="Read Article")})

# --- GLOSSARY FOOTER ---
with st.expander("System Glossary: Financial Metrics Explained"):
    st.markdown("""
    ### Valuation Ratios (Is the stock cheap or expensive?)
    * **P/E Ratio (Price-to-Earnings): The Price Tag on Profit**
        * This measures how much you are paying for every \$1 of profit the company makes.
        * **The Analogy:** If you buy a local business for \$100 and it makes \$10/year in profit, the P/E is 10x.
        * **High (>30x):** "Growth Mode." Investors are paying a premium price today because they expect massive profits tomorrow (e.g., AI or Tech stocks).
        * **Low (<15x):** "Value Mode." The stock is effectively "on sale," often because the industry is older or currently unloved (e.g., Banks or Energy).
    * **P/B Ratio (Price-to-Book): The Asset Test**
        * This compares the share price to the actual "Net Worth" of the company's hard assets (cash, factories, inventory).
        * **The Analogy:** If a company went bankrupt, sold all its factories, and paid off all debts, the Book Value is what would be left.
        * **< 1.0x:** "Deep Value." You are theoretically paying 80 cents to buy \$1.00 worth of hard assets. This is a rare signal that a stock is undervalued.
    ### AI Intelligence
    * **Sentiment Score:** The Real-Time Market Mood
        * **+1.0:** Maximum Optimism (Breaking good news).
        * **-1.0:** Maximum Pessimism (Crisis or panic).
        * **0.0:** Neutral (No significant news drivers).
    * **Divergence:** The Opportunity Gap
        * This metric calculates the difference between our Real-Time AI Score and the slow-moving Wall Street Consensus.
        * **Interpretation:** A large gap means the AI has detected breaking news that analysts have not yet factored into their quarterly ratings. This discrepancy represents potential alpha.
    """)
