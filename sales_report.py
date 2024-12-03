import streamlit as st
import pandas as pd
import plotly.express as px

# Streamlit 应用标题
st.set_page_config(page_title="广告趋势分析", layout="wide")
st.title("广告趋势分析面板")

# 文件上传组件
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# 定义页面选项
pages = ["数据预览", "趋势分析", "百分比分布", "趋势"]
selected_page = st.sidebar.selectbox("选择页面", pages)

# 上传文件后继续执行
if uploaded_file:
    # 读取数据
    df = pd.read_excel(uploaded_file, skiprows=3)

    # 日期列处理
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE']).dt.date  # 保留年月日

    # 数值型字段
    numeric_columns = ['TOTAL SALES', 'IMPRESSIONS', 'CLICKS', 'SPEND', 'CPC', 'CPM', 'CTR', 'ROAS']
    available_columns = [col for col in numeric_columns if col in df.columns]

    if selected_page == "数据预览":
        st.subheader("数据预览")
        st.write(df.head())

    elif selected_page == "趋势分析" and available_columns:
        st.subheader("趋势分析")
        # 用户选择分组字段
        aggregation_field = st.selectbox(
            "选择需要分析的参数:",
            options=available_columns,
            index=0
        )

        # 用户通过滑块选择排名范围
        max_campaigns = len(df['CAMPAIGN ID'].unique())
        min_rank, max_rank = st.slider(
            "选择Campaign IDs的排名范围:",
            min_value=1,
            max_value=max_campaigns,
            value=(1, min(5, max_campaigns)),
            step=1
        )

        # 汇总数据
        total_summary = (
            df.groupby('CAMPAIGN ID')[aggregation_field]
            .sum()
            .sort_values(ascending=False)
        )
        selected_campaigns = total_summary.iloc[min_rank - 1:max_rank].index

        sales_summary = (
            df[df['CAMPAIGN ID'].isin(selected_campaigns)]
            .groupby(['DATE', 'CAMPAIGN ID', 'PLACEMENT TYPE'])[aggregation_field]
            .sum()
            .reset_index()
        )

        # 创建折线图
        fig = px.line(
            sales_summary,
            x='DATE',
            y=aggregation_field,
            color='CAMPAIGN ID',
            line_dash='PLACEMENT TYPE',
            markers=True,
            title=f"Interactive Trend: {aggregation_field} by Campaign and Placement Type",
        )
        fig.update_layout(template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    elif selected_page == "百分比分布" and available_columns:
        st.subheader("百分比分布分析")
        aggregation_field = st.selectbox("选择需要分析的参数:", options=available_columns)

        # 用户选择日期范围
        date_range = st.date_input(
            "选择需要分析的时间段:",
            value=[df['DATE'].min(), df['DATE'].max()],
            min_value=df['DATE'].min(),
            max_value=df['DATE'].max(),
        )
        filtered_df = df[(df['DATE'] >= date_range[0]) & (df['DATE'] <= date_range[1])]

        # 用户选择具体日期
        available_dates = filtered_df['DATE'].unique()
        selected_date = st.selectbox("确定具体日期:", options=available_dates)

        # 筛选数据
        specific_date_data = filtered_df[filtered_df['DATE'] == selected_date]
        specific_date_summary = (
            specific_date_data.groupby('CAMPAIGN ID')[aggregation_field]
            .sum()
            .sort_values(ascending=False)
        )

        # 用户选择Top N范围
        max_campaigns_specific_date = len(specific_date_summary)
        top_n = st.slider(
            "选择展示前 N 个Campaign ID:",
            min_value=1,
            max_value=max_campaigns_specific_date,
            value=min(5, max_campaigns_specific_date),
        )

        # 创建百分比饼图
        top_n_campaigns = specific_date_summary.head(top_n)
        others = specific_date_summary.iloc[top_n:].sum()
        percentage_data = top_n_campaigns._append(pd.Series({'Others': others}))
        percentage_data = percentage_data / percentage_data.sum() * 100

        fig_pie = px.pie(
            names=percentage_data.index,
            values=percentage_data.values,
            title=f"{aggregation_field} 在 {selected_date} 的百分比分布分析",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.warning("请上传包含有效数据的文件以进行分析！")
else:
    st.info("请上传格式为.xlsx的Excel文件。")
