# 📈 AlphaStream Pro
**Real-Time Financial Intelligence Engine | Powered by GenAI & Snowflake**

### 🚀 [Launch Live App] (https://alphastream-pro-mohitvaid.streamlit.app/)|


📺 [Watch Demo]https://www.linkedin.com/posts/activity-7427116679391879168-gwyf?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAABsCPJkBE_P9p0ITMLkYDCeqOhdPD2Y9Eo8

![Dashboard Preview](assets/dashboard_screenshot.jpg)

## 📖 Executive Summary
AlphaStream Pro is an event-driven analytics platform designed to solve the "unstructured data" problem in finance. It ingests live news feeds, quantifies market sentiment using GPT-4, and persists data to Snowflake for historical trend analysis.

## 🏗️ Architecture
* **Ingestion:** Real-time Python ETL pipeline processing RSS feeds (Yahoo Finance).
* **AI Processing:** Integrated **OpenAI API (GPT-4)** for sentiment scoring (-1 to +1) and fuzzy matching on ticker symbols.
* **Storage:** **Snowflake Data Warehouse** for scalable, structured persistence.
* **Frontend:** **Streamlit** dashboard with forensic drill-down capabilities.

## 🛠️ Tech Stack
* **Core:** Python, Pandas, NumPy
* **Cloud & Data:** Snowflake, Azure
* **AI:** OpenAI API (LLM)
* **Visualization:** Streamlit, Plotly

## 📊 Key Features
1.  **Forensic Sentiment Tool:** Full data lineage from aggregate score down to specific source articles.
2.  **Alpha Hunter:** Scatter plot engine detecting divergence between News Sentiment and Price Action.
3.  **Automated ETL:** Replaces manual news reading with a 24/7 automated ingestion pipeline.
