import streamlit as st
import pandas as pd
import re
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# --- Configuration ---
st.set_page_config(page_title="Skincare Research Insights", layout="wide", page_icon="✨")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .css-1d391kg { background-color: #FFFFFF; padding: 2rem; border-radius: 10px;
                   box-shadow: 0 0 15px rgba(0,0,0,0.05); }
    h1, h2, h3, h4, h5, h6 { color: #007BFF; }
    .stButton>button { background-color: #28A745; color: black; border-radius: 5px; }
    .stSlider > div > div > div > div { background-color: #17A2B8; }
</style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    path = r"C:\Users\user\OneDrive\Desktop\data analysis project. final edits2\papers.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error("CSV file not found.")
        st.stop()
    df.columns = df.columns.str.strip().str.replace(' ', '').str.replace('[^A-Za-z0-9]+', '', regex=True)
    df['PTitle'] = df['PTitle'].astype(str)
    df['AuthorList'] = df['PAuthors'].apply(
        lambda x: [author.strip() for author in re.split(r'[;,]| and |\bandand\b', str(x)) if author.strip()]
    )
    if 'PYear' in df.columns:
        df['PYear'] = pd.to_numeric(df['PYear'], errors='coerce').fillna(0).astype(int)
    return df

df = load_data()

# --- Sidebar ---
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to:", ["📈 Trends", "👩‍🔬 Authors", "🌐 Co-authorship"])

# --- 1. Publication Trends ---
if section == "📈 Trends":
    st.header("🕰 Publication Trends")

    if 'PYear' in df.columns:
        papers_per_year = df['PYear'].value_counts().sort_index().reset_index()
        papers_per_year.columns = ['Year', 'Number_of_Papers']
        papers_per_year = papers_per_year[papers_per_year['Year'] > 0]

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(papers_per_year, x='Year', y='Number_of_Papers',
                         title="Total Papers Published Each Year",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            papers_per_year['Cumulative'] = papers_per_year['Number_of_Papers'].cumsum()
            fig2 = px.area(papers_per_year, x='Year', y='Cumulative',
                          title="Cumulative Publications",
                          color_discrete_sequence=[px.colors.qualitative.Pastel[2]])
            st.plotly_chart(fig2, use_container_width=True)

        if 'PType' in df.columns:
            type_data = df.groupby(['PYear', 'PType']).size().reset_index(name='Count')
            type_data = type_data[type_data['PYear'] > 0]
            fig3 = px.bar(type_data, x='PYear', y='Count', color='PType',
                          title="Publications by Type Over Time",
                          color_discrete_sequence=px.colors.qualitative.Light24)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("PYear column not found.")

# --- 2. Top Authors ---
elif section == "👩‍🔬 Authors":
    st.header("🤝 Top Authors")
    top_n = st.slider("Select number of top authors", 5, 30, 10)
    all_authors = df.explode('AuthorList')
    top_authors = all_authors['AuthorList'].value_counts().head(top_n).reset_index()
    top_authors.columns = ['Author', 'Publications']

    fig = px.bar(top_authors, x='Publications', y='Author', orientation='h',
                 title=f"Top {top_n} Authors",
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig, use_container_width=True)

# --- 3. Co-authorship Network ---
elif section == "🌐 Co-authorship":
    st.header("🌐 Co-authorship Network")

    if 'PID' not in df.columns:
        df['PID'] = df.index  # fallback

    G = nx.Graph()
    for _, row in df.iterrows():
        authors = row['AuthorList']
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a, b = authors[i], authors[j]
                if a != b:
                    if G.has_edge(a, b):
                        G[a][b]['weight'] += 1
                    else:
                        G.add_edge(a, b, weight=1)

    net = Network(height="700px", width="100%", notebook=False, heading='Co-authorship Graph')
    for node in G.nodes():
        net.add_node(node, label=node, size=10 + 2*G.degree(node))

    for source, target, data in G.edges(data=True):
        net.add_edge(source, target, value=data['weight'])

    net.save_graph("coauthors.html")
    HtmlFile = open("coauthors.html", 'r', encoding='utf-8')
    components.html(HtmlFile.read(), height=750)