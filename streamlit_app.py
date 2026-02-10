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

# --- 3. THE "LAZY LOADING" ENGINE (WITH HEARTBEAT FIX) ---
def run_live_update():
    if "openai" in st.secrets:
        try:
            # Check last update time
            try:
                last_update_df = session.sql("SELECT MAX(PUBLISH_DATE) as LAST_SYNC FROM FINANCE_DB.RAW_DATA.NEWS_SENTIMENT").to_pandas()
                last_sync = pd.to_datetime(last_update_df['LAST_SYNC'].iloc[0]) if not last_update_df.empty and last_update_df['LAST_SYNC'].iloc[0] else datetime.now() - timedelta(days=1)
            except:
                last_sync = datetime.now() - timedelta(days=1)
            
            # If data is older than 15 minutes, trigger refresh
            minutes_diff = (datetime.now() - last_sync).total_seconds() / 60
            
            if minutes_diff > 15:
                with st.spinner('⚡ Detecting Stale Data... AI Agent is fetching live global news...'):
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
                                articles.append({"TITLE": entry.title, "URL": entry.link, "PUBLISHED": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
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
                                # Insert News
                                session.sql(f"""
                                    INSERT INTO FINANCE_DB.RAW_DATA.NEWS_SENTIMENT (PUBLISH_DATE, TITLE, URL, TICKER, EVENT_TYPE, SENTIMENT_SCORE)
                                    SELECT '{art['PUBLISHED']}', '{art['TITLE'].replace("'", "''")}', '{art['URL']}', '{ticker}', '{event}', {score}
                                    WHERE NOT EXISTS (SELECT 1 FROM FINANCE_DB.RAW_DATA.NEWS_SENTIMENT WHERE TITLE = '{art['TITLE'].replace("'", "''")}')
                                """).collect()
                                new_tickers.append(ticker)
                        except: pass

                    # C. Update Prices (Safe Mode)
                    if new_tickers:
                        unique_tickers = list(set(new_tickers))
                        for symbol in unique_tickers:
                            try:
                                stock = yf.Ticker(symbol)
                                hist = stock.history(period="1d")
                                if not hist.empty:
                                    info = stock.info
                                    curr = hist['Close'].iloc[-1]
                                    open_p = hist['Open'].iloc[-1]
                                    chg = ((curr - open_p)/open_p)*100
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
                    
                    # D. THE HEARTBEAT FIX (Resets the clock!)
                    # We insert a dummy record to update MAX(PUBLISH_DATE) so the loop stops.
                    session.sql(f"""
                        INSERT INTO FINANCE_DB.RAW_DATA.NEWS_SENTIMENT (PUBLISH_DATE, TITLE, URL, TICKER, EVENT_TYPE, SENTIMENT_SCORE)
                        VALUES (CURRENT_TIMESTAMP(), 'System Heartbeat', 'N/A', 'SYSTEM', 'SYNC', 0.0)
                    """).collect()
                            
                    st.toast("✅ Live Data Sync Complete!", icon="🚀")
                    time.sleep(2)
                    st.rerun()
        except Exception as e:
            # Log error but don't crash
            print(f"Sync skipped: {e}")

# TRIGGER LIVE UPDATE
run_live_update()

# --- PROFESSIONAL CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f5f7f9; font-family: 'Inter', sans-serif; }
    div[data-baseweb="tab-list"] { gap: 8px; }
    button[data-baseweb="tab"] { background-color: #ffffff; border: 1px solid #e1e4e8; border-radius: 4px; color: #5e6c84; padding: 8px 16px; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: #e3f2fd; border: 1px solid #0052cc; color: #0052cc; border-bottom: 3px solid #0052cc; }
    .guide-card { background-color: #ebf3fc; padding: 12px 18px; border-radius: 6px; border-left: 4px solid #0052cc; margin-bottom: 20px; color: #172b4d; font-size: 0.85rem; }
    .methodology-card { background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #e1e4e8; height: 100%; }
    div[data-testid="stMetric"] { background-color: #ffffff; padding: 16px; border-radius: 8px; border: 1px solid #e1e4e8; border-left: 4px solid #0052cc; }
    .source-badge { background-color: #e3f2fd; color: #0d47a1; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; display: inline-block; margin-bottom: 10px; }
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
WHERE n.TICKER != 'SYSTEM' 
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
    last_update = pd.to_datetime(df['DB_NOW'].iloc[0]).strftime("%b %d, %I:%M %p UTC")
else:
    last_update = "Waiting for Data..."

# --- HEADER ---
c1, c2 = st.columns([6, 2])
with c1:
    st.title("AlphaStream Pro")
    st.markdown("**Real-Time Institutional Sentiment & Fundamental Intelligence** <span class='source-badge'>Universe: Event-Driven</span>", unsafe_allow_html=True)
with c2:
    if st.button("Refresh Data", type="primary"): st.rerun()
    st.markdown(f"<div style='text-align: right; color: #5e6c84; font-size: 0.85rem; margin-top: 5px;'>Last Sync: <b>{last_update}</b></div>", unsafe_allow_html=True)

st.divider()

# --- GLOBAL FILTER ---
col_search, col_space = st.columns([2, 2])
with col_search:
    all_tickers = sorted(df['TICKER'].unique().tolist()) if not df.empty else []
    selected_tickers = st.multiselect("Filter by Ticker", all_tickers)

display_df = df[df['TICKER'].isin(selected_tickers)] if selected_tickers else df

# --- KPI CARDS ---
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Live Articles Processed", len(display_df))
with k2: 
    sent = display_df['SENTIMENT_SCORE'].mean() if not display_df.empty else 0
    st.metric("Net Sentiment Score", f"{sent:.2f}", delta=f"{sent:.2f}")
with k3:
    if not selected_tickers:
        top = df.loc[df['CHANGE_PERCENT'].idxmax()] if not df.empty and 'CHANGE_PERCENT' in df.columns else None
        if top is not None:
            st.metric("Top Mover", f"{top['TICKER']} (${top['CURRENT_PRICE']:.2f})", f"{top['CHANGE_PERCENT']:.2f}%")
        else: st.metric("Top Mover", "-", "0%")
    else:
        avg_pe = display_df['PE_RATIO'].mean() if not display_df.empty else 0
        st.metric("Avg P/E Ratio", f"{avg_pe:.1f}x" if not pd.isna(avg_pe) else "N/A")
with k4: st.metric("Pipeline Status", "Active", delta="Live", delta_color="off")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Market Pulse", "Alpha Hunter", "Credibility Check"])

# === TAB 1: MARKET PULSE ===
with tab1:
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Sentiment Distribution")
        if not display_df.empty:
            display_df['Score Range'] = display_df['SENTIMENT_SCORE'].apply(lambda x: round(x * 5) / 5)
            hist_data = display_df.groupby('Score Range').size().reset_index(name='Volume')
            fig_hist = px.bar(hist_data, x="Score Range", y="Volume", color="Score Range", 
                color_continuous_scale=["#FF5252", "#E0E0E0", "#4CAF50"], range_color=[-1, 1], template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            with st.expander("Drill Down: Inspect Sentiment Drivers"):
                bucket = st.selectbox("Select Score Range:", options=sorted(display_df['Score Range'].unique()))
                subset = display_df[display_df['Score Range'] == bucket][['TICKER', 'TITLE', 'URL']]
                st.dataframe(subset, hide_index=True, use_container_width=True,
                    column_config={"URL": st.column_config.LinkColumn("Source", display_text="Read Article")})

    with c_right:
        st.subheader("Dominant Narratives")
        if not display_df.empty:
            theme_counts = display_df['EVENT_TYPE'].value_counts().reset_index()
            theme_counts.columns = ['Theme', 'Count']
            fig_donut = px.pie(theme_counts.head(7), values='Count', names='Theme', hole=0.6, color_discrete_sequence=px.colors.qualitative.G10)
            st.plotly_chart(fig_donut, use_container_width=True)

# === TAB 2: ALPHA HUNTER ===
with tab2:
    st.subheader("Price vs. Sentiment Correlation")
    if not display_df.empty:
        scatter_df = display_df.groupby('TICKER')[['SENTIMENT_SCORE', 'CHANGE_PERCENT', 'PE_RATIO']].agg(
            {'SENTIMENT_SCORE': 'mean', 'CHANGE_PERCENT': 'mean', 'PE_RATIO': 'max'}
        ).reset_index()
        
        fig_scatter = px.scatter(
            scatter_df, x="SENTIMENT_SCORE", y="CHANGE_PERCENT", text="TICKER", 
            template="plotly_white", size_max=60, hover_data=["PE_RATIO"]
        )
        fig_scatter.add_hline(y=0, line_dash="solid", line_color="#e1e4e8")
        fig_scatter.add_vline(x=0, line_dash="solid", line_color="#e1e4e8")
        st.plotly_chart(fig_scatter, use_container_width=True)

# === TAB 3: CREDIBILITY CHECK ===
with tab3:
    st.subheader("Analyst vs. AI Divergence")
    if not display_df.empty:
        comp_df = display_df.groupby('TICKER')[['SENTIMENT_SCORE', 'ANALYST_RATING']].agg(
            {'SENTIMENT_SCORE': 'mean', 'ANALYST_RATING': 'first'}
        ).reset_index()

        def rating_to_score(rating):
            r = str(rating).lower()
            if 'strong' in r: return 1.0
            if 'buy' in r: return 0.5
            if 'sell' in r: return -0.5
            return 0.0

        comp_df['Wall_St_Score'] = comp_df['ANALYST_RATING'].apply(rating_to_score)
        comp_df = comp_df.sort_values('SENTIMENT_SCORE', ascending=False).head(15)
        
        melted_df = comp_df.melt(id_vars=['TICKER'], value_vars=['SENTIMENT_SCORE', 'Wall_St_Score'], var_name='Source', value_name='Score')
        
        fig_side = px.bar(melted_df, x="TICKER", y="Score", color="Source", barmode='group',
             color_discrete_map={'SENTIMENT_SCORE': '#0052cc', 'Wall_St_Score': '#97a0af'})
        st.plotly_chart(fig_side, use_container_width=True)

# --- GLOSSARY ---
with st.expander("System Glossary"):
    st.markdown(r"""
    **P/E Ratio:** Price you pay for \$1 of earnings.
    **Sentiment:** AI Score from -1 (Negative) to +1 (Positive).
    """)
