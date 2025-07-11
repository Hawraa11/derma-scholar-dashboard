# 🧠 Research Intelligence Dashboard for Dermatology  
**Google Scholar Scraper + Forecasting + Visualization**

An end-to-end research analytics engine that scrapes, structures, and visualizes scholarly data from Google Scholar — focused on dermatology and inflammatory skin diseases. Features integrated forecasting, author insights, and citation network analysis.

---

## 📁 Project Components

### 🧰 Scrapers (Python)
- `researcher_scraper.py` — Scrapes researcher profile data (h-index, institution, etc.)
- `paper_scraper.py` — Scrapes publications by each researcher
- `citation_scraper.py` — Scrapes citations and references for each paper

### 📊 Datasets (CSV)
- `researchers.csv`
- `papers.csv`
- `citations.csv`

---

## 📈 Dashboard Features (Streamlit App)

### 🧑‍🔬 Researcher Insights
- Interactive researcher viewer
- h-index, i10-index, total publications & citations
- Affiliation breakdowns and filters

### 📚 Paper & Citation Analytics
- Total papers: **17,124**
- Total citations: **43,595**
- Publication distribution by year, type, and authorship

### 🔁 Time-Series Forecasting
- **Winter’s Exponential Smoothing**
- Seasonal + linear prediction models
- Forecasting research output & citations

### 🌐 Citation & Collaboration Network
- Co-authorship network graph
- Top researcher comparison
- Multi-modal citation forecasting

### 🌍 Institutional & Geographical Trends
- Top contributing institutions
- Country-wise publication breakdown

### 📊 Data Visualizations
- Research interest word clouds
- Productivity & citation bar charts
- Histograms, line plots, pie charts, seasonal trends

---

## 🛠 Tech Stack

- `Python`
- Scraping: `aiohttp`, `asyncio`, `requests`, `BeautifulSoup`
- Data: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`, `WordCloud`, `Streamlit`
- Forecasting: `statsmodels` (Winter’s model)
- Networks: `networkx`

---

## 🚀 Purpose & Impact

This dashboard was built to:
- Uncover dermatology research trends
- Discover top-performing researchers
- Understand citation dynamics and growth
- Predict future publication and citation patterns
- Serve as a template for other academic domains

---

## 🔗 Try It or Learn More

> Optionally add:
> - 🌐 **Live App Link** (if deployed)
> - 📽️ **Demo Video** or screenshots
> - 📘 **Related Blog Post** (if any)

---

## 🤝 Contributions & Future Work

Feel free to fork this project, suggest enhancements, or adapt it for other disciplines. Future plans:
- Add GPT-powered research summarization
- Enable filtering by disease/treatment keywords
- Enhance citation prediction with LSTM models
