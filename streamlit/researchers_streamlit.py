import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pycountry
from datetime import datetime
import os
import base64
import uuid

# ----------------------------
# SETUP & DATA LOADING
# ----------------------------

st.markdown("<h1 style='text-align: center;'>Dermatology Research Analytics </h1>", unsafe_allow_html=True)

# Function to load external CSV
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        
        # Data cleaning and type conversion
        numeric_cols = ['Total_Citations', 'Total_Publications', 'H_Index']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Generate yearly citation data if not present (for forecasting demo)
        if 'Years_Active' not in df.columns:
            df['Years_Active'] = np.random.randint(5, 25, size=len(df))
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return pd.DataFrame()

# File uploader
st.sidebar.title("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if not uploaded_file:
    st.warning("Please upload a CSV file to begin")
    st.stop()

df = load_data(uploaded_file)

# Show raw data option
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data Preview")
    st.dataframe(df)

# ----------------------------
# DATA ENHANCEMENTS
# ----------------------------

# Extract country from affiliation if not present
if 'Country' not in df.columns and 'Affiliation' in df.columns:
    df['Country'] = df['Affiliation'].str.extract(r'(\b[A-Za-z]+\b)$')[0]

# Create photo URLs if not present
if 'Photo_URL' not in df.columns and 'RName' in df.columns:
    df['Photo_URL'] = "https://ui-avatars.com/api/?name=" + df['RName'].str[0] + "&background=random"

# Generate yearly citation data for forecasting demo
for year in range(2018, 2024):
    col_name = f'Citations_{year}'
    if col_name not in df.columns and 'Total_Citations' in df.columns:
        df[col_name] = np.random.randint(
            df['Total_Citations']//10, 
            df['Total_Citations'], 
            size=len(df)
        )

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------

with st.sidebar:
    st.title("🔍 Dashboard Controls")
    
    # Personalization
    with st.expander("Personalization"):
        dark_mode = st.toggle("Dark Mode", False)
        st.session_state['watchlist'] = st.session_state.get('watchlist', [])
    
    # Data Filters
    with st.expander("Data Filters"):
        available_columns = df.columns.tolist()
        
        if 'Total_Citations' in available_columns:
            min_citations = st.slider(
                "Minimum Citations", 
                0, 
                int(df['Total_Citations'].max()), 
                0
            )
        
        if 'H_Index' in available_columns:
            min_h_index = st.slider(
                "Minimum H-Index", 
                0, 
                int(df['H_Index'].max()), 
                0
            )
        
        if 'Country' in available_columns:
            countries = st.multiselect(
                "Countries", 
                df['Country'].unique()
            )
        
        if 'Interests' in available_columns:
            interests_filter = st.text_input("Search by Research Interest")

# Apply filters
filtered_df = df.copy()
if 'Total_Citations' in available_columns and min_citations > 0:
    filtered_df = filtered_df[filtered_df['Total_Citations'] >= min_citations]
if 'H_Index' in available_columns and min_h_index > 0:
    filtered_df = filtered_df[filtered_df['H_Index'] >= min_h_index]
if 'Country' in available_columns and countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
if 'Interests' in available_columns and interests_filter:
    filtered_df = filtered_df[filtered_df['Interests'].str.contains(
        interests_filter, case=False, na=False
    )]

# ----------------------------
# MAIN DASHBOARD
# ----------------------------


# ----------------------------
# SECTION 1: KEY METRICS
# ----------------------------

st.header("📊 Executive Summary")

cols = st.columns(4)
if 'RName' in available_columns:
    cols[0].metric("Total Researchers", len(filtered_df))
if 'Total_Citations' in available_columns:
    cols[1].metric("Avg Citations", f"{filtered_df['Total_Citations'].mean():,.0f}")
if 'H_Index' in available_columns:
    cols[2].metric("Avg H-Index", round(filtered_df['H_Index'].mean(), 1))
if 'Country' in available_columns:
    cols[3].metric("Int'l Diversity", f"{len(filtered_df['Country'].unique())} countries")

# ----------------------------
# SECTION 2: ADVANCED ANALYTICS
# ----------------------------

st.header("🔍 Advanced Analytics")

tab1, tab2, tab3 = st.tabs(["Collaboration Network", "Impact Scoring", "Benchmarking"])

with tab1:
    st.subheader("Potential Collaboration Network")
    
    if 'RName' in available_columns and 'Interests' in available_columns:
        # Create network graph
        G = nx.Graph()
        for _, row in filtered_df.iterrows():
            G.add_node(row['RName'], size=row.get('Total_Citations', 0)/500)
        
      # Add edges based on shared specialties
for i in range(len(filtered_df)):
    interests_i = filtered_df.iloc[i]['Interests']
    name_i = filtered_df.iloc[i]['RName']
    if not isinstance(interests_i, str):
        continue

    set_i = set(interests_i.split(', '))

    for j in range(i+1, len(filtered_df)):
        interests_j = filtered_df.iloc[j]['Interests']
        name_j = filtered_df.iloc[j]['RName']
        if not isinstance(interests_j, str):
            continue

        set_j = set(interests_j.split(', '))
        common_interests = set_i & set_j

        if common_interests:
            G.add_edge(name_i, name_j, weight=len(common_interests))

        
        # Visualize with Plotly
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x, node_y, node_text, node_size = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_size.append(G.nodes[node]['size']*10)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_size,
                colorscale='Viridis',
                showscale=True
            ),
            hoverinfo='text'
        )
        
    if not G.edges:
      st.warning("No connections found based on shared interests.")
    else:
     fig = go.Figure(data=[edge_trace, node_trace])
     unique_key = f"network_graph_{uuid.uuid4().hex}"
     st.plotly_chart(fig, use_container_width=True, key=unique_key)

with tab2:
    st.subheader("Custom Impact Scoring")
    
    if {'Total_Citations', 'H_Index', 'Total_Publications'}.issubset(available_columns):
    # Ensure at least one row has non-null values in those columns
      subset_df = filtered_df[['Total_Citations', 'H_Index', 'Total_Publications']].dropna()
      if len(subset_df) == 0:
        st.warning("No valid data found for Impact Scoring (all NaNs or empty).")
      else:
        col1, col2, col3 = st.columns(3)
        cite_weight = col1.slider("Citation Weight", 0, 100, 50)
        h_weight = col2.slider("H-Index Weight", 0, 100, 30)
        pub_weight = col3.slider("Publication Weight", 0, 100, 20)

        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(subset_df)
        weights = np.array([cite_weight, h_weight, pub_weight]) / 100

        # Align filtered_df index with subset_df for assigning scores correctly
        scores = (normalized * weights).sum(axis=1)
        filtered_df.loc[subset_df.index, 'Impact_Score'] = scores

        st.dataframe(
            filtered_df.sort_values('Impact_Score', ascending=False)[
                ['RName', 'Impact_Score', 'Total_Citations', 'H_Index', 'Total_Publications']
            ].style.background_gradient(cmap='Blues'),
            height=400
        )
    else:
      st.warning("Impact scoring requires citation, h-index, and publication data")

with tab3:
    st.subheader("Researcher Benchmarking")
    
if 'RName' in available_columns:
    selected = st.multiselect(
        "Compare researchers", 
        filtered_df['RName'], 
        default=filtered_df['RName'][:2],
        key="compare_researchers_unique"  # unique key here
    )
    
    if len(selected) >= 2:
        compare_df = filtered_df[filtered_df['RName'].isin(selected)]
        
        metrics = []
        if 'Total_Citations' in available_columns: metrics.append('Total_Citations')
        if 'H_Index' in available_columns: metrics.append('H_Index')
        if 'Total_Publications' in available_columns: metrics.append('Total_Publications')
        if 'Years_Active' in available_columns: metrics.append('Years_Active')

        if len(metrics) >= 3:  # Need at least 3 metrics for radar chart
            scaled = MinMaxScaler().fit_transform(compare_df[metrics])
            
            fig = go.Figure()
            for idx in range(len(compare_df)):
                fig.add_trace(go.Scatterpolar(
                    r=np.append(scaled[idx], scaled[idx][0]),
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=compare_df.iloc[idx]['RName']
                ))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 3 metrics for radar chart")
else:
    st.warning("Benchmarking requires researcher names")

# SECTION 3: FORECASTING TOOLS
# ----------------------------

st.header("📈 Forecasting Tools")

forecast_tab1, forecast_tab2 = st.tabs(["Citation Projections", "Field Growth Rates"])

with forecast_tab1:
    st.subheader("Multi-Model Citation Forecasting")
    
if 'RName' in available_columns:
    # Clean up RName column just in case
    filtered_df = filtered_df.dropna(subset=['RName'])
    filtered_df['RName'] = filtered_df['RName'].astype(str).str.strip()

    selected_researcher = st.selectbox("Select researcher", filtered_df['RName'].unique())
    model_type = st.selectbox("Growth model", ["Linear", "Exponential", "Logistic"])
    years = st.slider("Projection years", 1, 10, 5)
    
    # Extract citation columns
    citation_cols = [col for col in filtered_df.columns if col.startswith('Citations_')]
    
    researcher_row = filtered_df[filtered_df['RName'] == selected_researcher]
    
    if not researcher_row.empty:
        if citation_cols:
            history = researcher_row[citation_cols].values.flatten()
            years_history = [int(col.split('_')[1]) for col in citation_cols]
        elif 'Total_Citations' in available_columns:
            history = [researcher_row['Total_Citations'].values[0]]
            years_history = [2023]  # fallback year
        else:
            st.warning("No citation history found.")
            st.stop()

        # Project future citations
        future_years = list(range(max(years_history)+1, max(years_history)+years+1))
        
        if model_type == "Linear":
            growth = (history[-1] - history[0]) / len(history) if len(history) > 1 else history[0]*0.1
            forecast = [history[-1] + growth*i for i in range(1, years+1)]
        elif model_type == "Exponential":
          growth = (history[-1] / history[0])**(1 / len(history)) - 1 if len(history) > 1 else 0.1
          forecast = [history[-1] * (1 + growth)**i for i in range(1, years + 1)]

        else:  # Logistic
            cap = history[-1] * 1.5
            forecast = [cap / (1 + np.exp(-0.5*(i - years/2))) for i in range(1, years+1)]

        # Plot the result
        fig = px.line(
            x=years_history + future_years,
            y=list(history) + forecast,
            labels={'x': 'Year', 'y': 'Citations'},
            title=f"{model_type} Projection for {selected_researcher}"
        )
        fig.add_vline(x=max(years_history)+0.5, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data found for {selected_researcher}")
else:
    st.warning("Forecasting requires researcher names")

with forecast_tab2:
   st.subheader("Specialty Growth Rates")

if 'Interests' in available_columns:
    # Automatically detect specialties from interests
    all_interests = filtered_df['Interests'].dropna().str.split(', ').explode().value_counts()
    top_specialties = all_interests.head(5).index.tolist()

    if top_specialties:
        specialty_rates = {spec: np.random.uniform(0.03, 0.1) for spec in top_specialties}
        
        st.dataframe(pd.DataFrame.from_dict(specialty_rates, orient='index', columns=['Growth Rate']))

        selected_specialty = st.selectbox("View researchers in", list(specialty_rates.keys()))

        if selected_specialty:
            projected_growth = (1 + specialty_rates[selected_specialty])**5 - 1
            st.write(f"📈 Projected 5-year growth: *{projected_growth:.0%}*")

            # Filter researchers
            specialty_df = filtered_df[
                filtered_df['Interests'].notna() & 
                filtered_df['Interests'].str.contains(selected_specialty, na=False)
            ]

            if not specialty_df.empty:
                cols_to_show = ['RName']
                if 'Total_Citations' in available_columns: cols_to_show.append('Total_Citations')
                if 'H_Index' in available_columns: cols_to_show.append('H_Index')
                st.dataframe(specialty_df[cols_to_show])
            else:
                st.warning("No researchers found in this specialty.")
        else:
            st.warning("Please select a specialty to view growth.")
    else:
        st.warning("No specialties found in the dataset.")
else:
    st.warning("Specialty analysis requires research interests data.")
# ----------------------------
# SECTION 4: GEO & INSTITUTIONAL ANALYSIS
# ----------------------------

st.header("🌐 Geographic & Institutional Insights")

if 'Country' in available_columns:
    # Geocoding
    try:
        country_codes = {country.name: country.alpha_3 for country in pycountry.countries}
        filtered_df['Country_Code'] = filtered_df['Country'].map(country_codes)
    except:
        filtered_df['Country_Code'] = 'USA'  # Fallback
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Global Research Distribution")
        country_counts = filtered_df.groupby(['Country', 'Country_Code']).size().reset_index(name='Count')
        if not country_counts.empty:
            fig = px.choropleth(
                country_counts,
                locations="Country_Code",
                color="Count",
                hover_name="Country",
                projection="natural earth"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Institutions")
        if 'Affiliation' in available_columns:
            inst_df = filtered_df.groupby('Affiliation').agg({
                'Total_Citations': 'mean',
                'H_Index': 'mean',
                'RName': 'count'
            }).rename(columns={'RName': 'Researchers'}).sort_values('Total_Citations', ascending=False)
            
            if not inst_df.empty:
                fig = px.bar(
                    inst_df.head(10),
                    x=inst_df.head(10).index,
                    y='Total_Citations',
                    color='H_Index',
                    labels={'x': 'Institution', 'Total_Citations': 'Avg Citations'}
                )
                st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# SECTION 5: RESEARCH TOPICS & TRENDS
# ----------------------------

st.header("🔬 Research Topic Analysis")

if 'Interests' in available_columns:
    # Topic Modeling
    st.subheader("Emerging Research Themes")
    try:
        vectorizer = CountVectorizer(max_df=0.9, min_df=1, stop_words='english')
        dtm = vectorizer.fit_transform(filtered_df['Interests'])
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(dtm)
        
        for idx, topic in enumerate(lda.components_):
            st.write(f"Topic #{idx+1}:")
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
            st.write(", ".join(top_words))
    except:
        st.warning("Could not perform topic modeling - not enough unique interests")
    
        st.subheader("Research Interest Word Cloud")

              # Combine all interests into one string, excluding NaNs and empty strings
        interests_series = filtered_df['Interests'].dropna().str.strip()
        text = ' '.join(interests_series[interests_series != ''])

        if text.strip():  # Ensure there's at least one non-empty word
              wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
              st.image(wordcloud.to_array())
        else:     
          st.warning("No valid research interests available to generate a word cloud.")

else:
    st.warning("Topic analysis requires research interests data")

# ----------------------------
# SECTION 6: RESEARCHER PROFILES
# ----------------------------

st.header("🧑‍⚕ Researcher Profiles")

if 'RName' in available_columns:
    # Masonry grid layout
    cols = st.columns(3)
    for idx, (_, row) in enumerate(filtered_df.iterrows()):
        with cols[idx % 3]:
            with st.expander(f"{row['RName']}"):
                if 'Photo_URL' in available_columns:
                    st.image(row['Photo_URL'], width=150)
                if 'Affiliation' in available_columns:
                    st.write(f"*Affiliation:* {row['Affiliation']}")
                if 'Total_Citations' in available_columns:
                    st.write(f"*Citations:* {row['Total_Citations']:,}")
                if 'H_Index' in available_columns:
                    st.write(f"*H-Index:* {row['H_Index']}")
                
                if st.button("Add to Watchlist", key=f"watch_{row['RName']}"):
                    st.session_state['watchlist'].append(row['RName'])
                
                if st.button("Find Similar", key=f"sim_{row['RName']}"):
                    if 'Interests' in available_columns:
                        similar = filtered_df[
                            filtered_df['Interests'].apply(
                                lambda x: len(set(str(x).split(', ')) & set(str(row['Interests']).split(', '))) > 1
                            )
                        ]
                        st.dataframe(similar[['RName', 'Interests']])

    # Watchlist display
    if st.session_state['watchlist']:
        st.subheader("⭐ Your Watchlist")
        st.write(", ".join(st.session_state['watchlist']))
else:
    st.warning("Researcher profiles require names")

# ----------------------------
# SECTION 7: BONUS ANALYTICS
# ----------------------------

st.header("💡 Bonus Insights")

tab1, tab2, tab3 = st.tabs(["Funding Potential", "Publication Velocity", "Domain Share"])

with tab1:
    st.subheader("Grant Funding Estimator")
    if 'Impact_Score' in filtered_df.columns:
        filtered_df['Funding_Estimate'] = filtered_df['Impact_Score'] * 500000  # Mock calculation
        cols_to_show = ['RName', 'Funding_Estimate']
        if 'Affiliation' in available_columns:
            cols_to_show.append('Affiliation')
        st.dataframe(filtered_df[cols_to_show].sort_values('Funding_Estimate', ascending=False))

with tab2:
    st.subheader("Publication Velocity")
    if 'Total_Publications' in available_columns and 'Years_Active' in available_columns:
        filtered_df['Papers_Per_Year'] = filtered_df['Total_Publications'] / filtered_df['Years_Active']
        cols_to_show = ['RName', 'Papers_Per_Year']
        if 'Affiliation' in available_columns:
            cols_to_show.append('Affiliation')
        st.dataframe(filtered_df[cols_to_show].sort_values('Papers_Per_Year', ascending=False))

with tab3:
    st.subheader("Research Domain Share")
    if 'Interests' in available_columns:
        domain_counts = filtered_df['Interests'].str.split(', ').explode().value_counts()
        if not domain_counts.empty:
            fig = px.pie(domain_counts, values=domain_counts.values, names=domain_counts.index)
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# FOOTER & DATA EXPORT
# ----------------------------

st.download_button(
    label="📥 Download Full Analysis",
    data=filtered_df.to_csv(index=False),
    file_name="dermatology_research_analytics.csv",
    mime="text/csv"
)

st.caption(f"Analytics Dashboard v2.0 | Data updated: {datetime.now().strftime('%Y-%m-%d')}")