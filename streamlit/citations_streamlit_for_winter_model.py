import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from wordcloud import WordCloud
from collections import Counter
import altair as alt
from datetime import datetime
import plotly.graph_objects as go

# Set winter theme with enhanced styling
st.set_page_config(
    page_title="Advanced Citation Forecasting",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for winter theme
st.markdown("""
<style>
    :root {
        --primary-color: #0066cc;
        --secondary-color: #e6f2ff;
        --accent-color: #4da6ff;
    }
    .stApp {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f2ff 100%);
        color: #003366;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #e6f2ff 0%, #cce5ff 100%);
        border-right: 1px solid #b3d9ff;
    }
    h1, h2, h3 {
        color: var(--primary-color);
        border-bottom: 1px solid var(--accent-color);
        padding-bottom: 0.3rem;
    }
    .st-bq {
        border-left: 3px solid var(--accent-color);
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 5px;
    }
    .stSelectbox, .stSlider, .stTextInput {
        background-color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .snowflake {
        color: var(--accent-color);
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced data loading with caching and validation
@st.cache_data(show_spinner="Loading and processing your data...")
def load_and_validate_data(uploaded_file):
    try:
        # Try multiple encodings if needed
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        # Check required columns
        required_columns = {'CID', 'PID', 'CTitle', 'CAuthors', 'CYear', 'CPublisher'}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Data cleaning
        df = df.dropna(subset=['CYear'])
        df['CYear'] = pd.to_numeric(df['CYear'], errors='coerce')
        df = df.dropna(subset=['CYear'])
        df['CYear'] = df['CYear'].astype(int)
        
        # Extract decade for additional analysis
        df['Decade'] = (df['CYear'] // 10) * 10
        
        # Process authors
        df['Author_Count'] = df['CAuthors'].apply(lambda x: len(str(x).split(',')))
        df['First_Author'] = df['CAuthors'].apply(lambda x: str(x).split(',')[0].strip())
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Enhanced forecasting function
def run_forecasting(ts_data, forecast_years, trend_type, seasonal_type, seasonal_period):
    try:
        # Split into train and test sets (last 20% for validation)
        split_idx = int(len(ts_data) * 0.8)
        train = ts_data.iloc[:split_idx]
        test = ts_data.iloc[split_idx:] if split_idx < len(ts_data) else None
        
        # Fit model
        model = ExponentialSmoothing(
            train['Citations'],
            trend=trend_type,
            seasonal=seasonal_type,
            seasonal_periods=seasonal_period,
            initialization_method='estimated'
        ).fit()
        
        # Generate predictions
        forecast = model.forecast(forecast_years)
        forecast_years = np.arange(ts_data.index[-1]+1, ts_data.index[-1]+1+forecast_years)
        forecast_df = pd.DataFrame({'Year': forecast_years, 'Forecast': forecast}).set_index('Year')
        
        # Calculate metrics if test set exists
        metrics = {}
        if test is not None and len(test) > 0:
            pred_test = model.forecast(len(test))
            metrics['MAE'] = mean_absolute_error(test['Citations'], pred_test)
            metrics['RMSE'] = np.sqrt(mean_squared_error(test['Citations'], pred_test))
            metrics['MAPE'] = np.mean(np.abs((test['Citations'] - pred_test) / test['Citations'])) * 100
        
        return model, forecast_df, metrics
    
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")
        return None, None, None

# Main app
def main():
    st.title("❄️ Citation Analytics & Forecasting")
    st.markdown("""
    <div style="background-color: #e6f2ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    Analyze citation trends, author networks, and forecast future impact using advanced time series modeling.
    </div>
    """, unsafe_allow_html=True)
    
    # File upload with better UX
    with st.sidebar:
        st.markdown("<div class='snowflake'></div>", unsafe_allow_html=True)
        st.subheader("Data Input")
        uploaded_file = st.file_uploader("Upload citation data (CSV)", type=["csv"])
        
        if uploaded_file is None:
            sample_data = st.checkbox("Use sample data for demo")
            if sample_data:
                data = {
                    'CID': ['52b96692d5a1c62813dea0d6a3abf402db7e1a5a', '146f28fcbd59332d3c3a2babf6ef97fd39567cd8', '01905a85d3f507df1561a8c00bd7a025c84f8ceb'],
                    'PID': [12, 246, 717],
                    'CTitle': ['Comparative analysis of multiple-casualty incident triage algorithms.', 
                              'Feasibility of transcranial Doppler to evaluate vasculopathy among survivors of childhood brain tumors exposed to cranial radiation therapy',
                              'Mental wellness and health-related quality of life of young adult survivors of childhood cancer in Singapore.'],
                    'CAuthors': ['A. Garner, Anna Lee, Ken Harrison, C. Schultz', 
                                'Daniel C Bowers, Mark D Johnson',
                                'Francis Jia Yi Fong, Bryan Wei Zhi Wong, Jamie Si Pin Ong, B. Tan, Michaela Su-Fern Seng, Ah-Moy Tan, R. Tanugroho'],
                    'CYear': [2001, 2024, 2024],
                    'CPublisher': ['Annals of Emergency Medicine', 'Pediatric Blood & Cancer', 'Annals of the Academy of Medicine, Singapore']
                }
                df = pd.DataFrame(data)
                df['Decade'] = (df['CYear'] // 10) * 10
                df['Author_Count'] = df['CAuthors'].apply(lambda x: len(str(x).split(',')))
                df['First_Author'] = df['CAuthors'].apply(lambda x: str(x).split(',')[0].strip())
            else:
                st.warning("Please upload a CSV file or use sample data")
                st.stop()
        else:
            df = load_and_validate_data(uploaded_file)
            if df is None or df.empty:
                st.error("Invalid data format. Please check your CSV file.")
                st.stop()
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Forecasting", "👥 Authors", "🔍 Deep Dive"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='metric-card'>📅 <b>Time Span</b><br>{} - {}</div>".format(
                df['CYear'].min(), df['CYear'].max()), unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'>📚 <b>Publications</b><br>{:,}</div>".format(
                len(df)), unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'>👥 <b>Unique Authors</b><br>{:,}</div>".format(
                len(set(','.join(df['CAuthors'].dropna()).split(',')))), unsafe_allow_html=True)
        
        # Time distribution
        st.subheader("Publication Timeline")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.histplot(data=df, x='CYear', bins=30, kde=True, color='#4da6ff', ax=ax1)
        ax1.set_title('Distribution of Publications by Year', fontsize=14)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Publications')
        st.pyplot(fig1)
        
        # Publisher analysis
        st.subheader("Publisher Analysis")
        top_publishers = df['CPublisher'].value_counts().head(10)
        fig2 = go.Figure(go.Bar(
            x=top_publishers.values,
            y=top_publishers.index,
            orientation='h',
            marker_color='#0066cc'
        ))
        fig2.update_layout(
            title='Top 10 Publishers by Publication Count',
            xaxis_title='Number of Publications',
            yaxis_title='Publisher',
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.header("Citation Forecasting")
        
        # Prepare time series data
        citation_counts = df['CYear'].value_counts().sort_index()
        ts_df = pd.DataFrame({
            'Year': citation_counts.index,
            'Citations': citation_counts.values
        }).set_index('Year')
        
        # Forecasting parameters
        with st.expander("⚙️ Forecasting Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                forecast_years = st.slider("Years to forecast", 1, 10, 3)
            with col2:
                seasonal_period = st.slider("Seasonal period", 1, 10, 5)
            with col3:
                model_type = st.selectbox("Model type", 
                                         ["Additive", "Multiplicative"])
        
        trend_type = 'add' if model_type == "Additive" else 'mul'
        seasonal_type = 'add' if model_type == "Additive" else 'mul'
        
        # Run forecasting
        model, forecast_df, metrics = run_forecasting(
            ts_df, forecast_years, trend_type, seasonal_type, seasonal_period)
        
        if model and forecast_df is not None:
            # Plot results
            fig3 = go.Figure()
            
            # Actual data
            fig3.add_trace(go.Scatter(
                x=ts_df.index,
                y=ts_df['Citations'],
                name='Actual',
                line=dict(color='#0066cc', width=3),
                mode='lines+markers'
            ))
            
            # Forecast
            fig3.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Forecast'],
                name='Forecast',
                line=dict(color='#ff6666', width=3, dash='dash'),
                mode='lines+markers'
            ))
            
            # Confidence interval
            fig3.add_trace(go.Scatter(
                x=np.concatenate([forecast_df.index, forecast_df.index[::-1]]),
                y=np.concatenate([
                    forecast_df['Forecast'] * 0.8,
                    forecast_df['Forecast'] * 1.2
                ][::-1]),
                fill='toself',
                fillcolor='rgba(255,102,102,0.2)',
                line_color='rgba(255,255,255,0)',
                name='Confidence Interval'
            ))
            
            fig3.update_layout(
                title='Citation Forecast with Holt-Winters Model',
                xaxis_title='Year',
                yaxis_title='Number of Citations',
                hovermode='x unified',
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Show metrics
            if metrics:
                st.subheader("Model Performance")
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
                m2.metric("Root Mean Squared Error", f"{metrics['RMSE']:.2f}")
                m3.metric("Mean Absolute Percentage Error", f"{metrics['MAPE']:.2f}%")
            
            # Show forecast table
            st.subheader("Forecast Details")
            forecast_df['Forecast'] = forecast_df['Forecast'].round(1)
            st.dataframe(forecast_df.style.background_gradient(cmap='Blues'))
    
    with tab3:
        st.header("Author Network Analysis")
        
        # Process authors data
        all_authors = []
        for authors in df['CAuthors'].dropna():
            all_authors.extend([a.strip() for a in authors.split(',')])
        
        author_counts = pd.Series(all_authors).value_counts().head(20)
        
        # Top authors
        st.subheader("Most Prolific Authors")
        fig4 = go.Figure(go.Bar(
            x=author_counts.values,
            y=author_counts.index,
            orientation='h',
            marker_color='#4da6ff'
        ))
        fig4.update_layout(
            height=600,
            title='Top 20 Authors by Publication Count'
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        # Collaboration network
        st.subheader("Collaboration Patterns")
        st.info("Author network visualization would appear here with a larger dataset", icon="ℹ️")
        
        # Author trends over time
        st.subheader("Author Productivity Over Time")
        author_years = []
        for _, row in df.iterrows():
            for author in row['CAuthors'].split(','):
                author_years.append((author.strip(), row['CYear']))
        
        author_trends = pd.DataFrame(author_years, columns=['Author', 'Year'])
        top_5_authors = author_trends['Author'].value_counts().head(5).index
        
        fig5 = alt.Chart(
            author_trends[author_trends['Author'].isin(top_5_authors)]
        ).mark_line(point=True).encode(
            x='Year:O',
            y='count():Q',
            color='Author:N',
            tooltip=['Author', 'count()']
        ).properties(
            height=500,
            title='Publication Trends for Top 5 Authors'
        ).interactive()
        
        st.altair_chart(fig5, use_container_width=True)
    
    with tab4:
        st.header("Deep Dive Analysis")
        
        # Word cloud of titles
        st.subheader("Title Word Cloud")
        title_text = ' '.join(df['CTitle'].dropna().astype(str))
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='winter',
            max_words=100
        ).generate(title_text)
        
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        ax6.imshow(wordcloud, interpolation='bilinear')
        ax6.axis('off')
        ax6.set_title('Most Frequent Words in Publication Titles', fontsize=16)
        st.pyplot(fig6)
        
        # Author count analysis
        st.subheader("Collaboration Size Analysis")
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Author_Count'], bins=15, kde=True, color='#0066cc', ax=ax7)
        ax7.set_title('Distribution of Authors per Publication', fontsize=14)
        ax7.set_xlabel('Number of Authors')
        ax7.set_ylabel('Count')
        st.pyplot(fig7)
        
        # Temporal trends by decade
        st.subheader("Historical Trends by Decade")
        decade_counts = df['Decade'].value_counts().sort_index()
        fig8 = go.Figure(go.Bar(
            x=decade_counts.index,
            y=decade_counts.values,
            marker_color='#4da6ff'
        ))
        fig8.update_layout(
            title='Publications by Decade',
            xaxis_title='Decade',
            yaxis_title='Number of Publications',
            height=500
        )
        st.plotly_chart(fig8, use_container_width=True)

if __name__ == "__main__":
    main()