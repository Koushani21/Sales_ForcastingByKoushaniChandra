import streamlit as st
import pandas as pd
import plotly.express as px


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Retail Sales Forecasting",
    layout="wide"
)


# -------------------------------------------------
# Load Data
# -------------------------------------------------

df = pd.read_csv("train.csv")

forecast_df = pd.read_csv("forecast.csv")

anomaly_df = pd.read_csv("anomalies.csv")

cluster_df = pd.read_csv("clusters.csv")



# -------------------------------------------------
# Data Preparation
# -------------------------------------------------

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="mixed",
    dayfirst=True
)

df["Year"] = df["Order Date"].dt.year

df["Month"] = (
    df["Order Date"]
    .dt.to_period("M")
    .astype(str)
)



# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------

st.sidebar.title(
    "Retail Forecasting Dashboard"
)


page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Overview Dashboard",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)



# =================================================
# PAGE 1 : SALES OVERVIEW
# =================================================

if page == "Sales Overview Dashboard":

    st.title(
        "Sales Overview Dashboard"
    )


    yearly_sales = (
        df.groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )


    fig1 = px.bar(
        yearly_sales,
        x="Year",
        y="Sales",
        title="Total Sales by Year"
    )


    st.plotly_chart(
        fig1,
        use_container_width=True
    )



    monthly_sales = (
        df.groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )


    fig2 = px.line(
        monthly_sales,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )



    st.subheader(
        "Sales by Region and Category"
    )


    region = st.selectbox(
        "Select Region",
        df["Region"].unique()
    )


    category = st.selectbox(
        "Select Category",
        df["Category"].unique()
    )


    filtered = df[
        (df["Region"] == region)
        &
        (df["Category"] == category)
    ]


    fig3 = px.bar(
        filtered,
        x="Sub-Category",
        y="Sales",
        title="Sales Distribution"
    )


    st.plotly_chart(
        fig3,
        use_container_width=True
    )





# =================================================
# PAGE 2 : FORECAST EXPLORER
# =================================================

elif page == "Forecast Explorer":

    st.title(
        "XGBoost Forecast Explorer"
    )


    forecast_type = st.selectbox(
        "Forecast By",
        [
            "Category",
            "Region"
        ]
    )


    # Category selection

    if forecast_type == "Category":

        selected_value = st.selectbox(
            "Select Category",
            df["Category"].unique()
        )


        if "Category" in forecast_df.columns:

            filtered_forecast = forecast_df[
                forecast_df["Category"] == selected_value
            ]

        else:

            filtered_forecast = forecast_df



    # Region selection

    else:

        selected_value = st.selectbox(
            "Select Region",
            df["Region"].unique()
        )


        if "Region" in forecast_df.columns:

            filtered_forecast = forecast_df[
                forecast_df["Region"] == selected_value
            ]

        else:

            filtered_forecast = forecast_df



    horizon = st.slider(
        "Forecast Horizon (Months Ahead)",
        min_value=1,
        max_value=3,
        value=1
    )



    forecast_result = filtered_forecast.head(
        horizon
    )



    fig = px.line(
        forecast_result,
        x="Month",
        y="Forecast",
        markers=True,
        title=f"XGBoost Forecast - {selected_value}"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



    st.subheader(
        "Forecast Output"
    )


    st.dataframe(
        forecast_result
    )



    st.subheader(
        "XGBoost Model Evaluation"
    )


    col1, col2 = st.columns(2)


    col1.metric(
        "MAE",
        14763.81
    )


    col2.metric(
        "RMSE",
        18337.41
    )





# =================================================
# PAGE 3 : ANOMALY REPORT
# =================================================

elif page == "Anomaly Report":

    st.title(
        "Sales Anomaly Detection"
    )


    anomaly_df["Order Date"] = pd.to_datetime(
        anomaly_df["Order Date"],
        format="mixed",
        dayfirst=True
    )


    fig = px.line(
        anomaly_df,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Sales Trend with Anomalies"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



    st.subheader(
        "Detected Anomaly Dates"
    )


    anomalies = anomaly_df[
        (anomaly_df["IF_Anomaly"] == 0)
        |
        (anomaly_df["Z_Anomaly"] == 1)
    ]


    st.dataframe(
        anomalies[
            [
                "Order Date",
                "Sales",
                "IF_Anomaly",
                "Z_Anomaly"
            ]
        ]
    )


    st.subheader(
        "Detection Summary"
    )


    c1,c2,c3 = st.columns(3)


    c1.metric(
        "Detected by Both",
        1
    )


    c2.metric(
        "Only Isolation Forest",
        10
    )


    c3.metric(
        "Only Z-Score",
        5
    )





# =================================================
# PAGE 4 : PRODUCT DEMAND SEGMENTS
# =================================================

elif page == "Product Demand Segments":

    st.title(
        "Product Demand Clusters"
    )


    st.subheader(
        "Demand Cluster Visualization"
    )


    st.image(
        "clusterimage.jpeg",
        caption="Product Demand Clusters",
        use_container_width=True
    )


    st.subheader(
        "Sub-Category Cluster Mapping"
    )


    st.dataframe(
        cluster_df
    )