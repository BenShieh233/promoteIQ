import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    units_mapping = {
        'TOTAL SALES': 'USD',
        'IMPRESSIONS': 'Count',
        'CLICKS': 'Count',
        'SPEND': 'USD',
        'CPC': 'USD',
        'CPM': 'USD',
        'CTR': 'Percentage (%)',
        'ROAS': 'Ratio'
    }

    if selected_page == "数据预览":
        st.subheader("数据预览")

        # Section 1: 数据样本
        st.write("##### 数据样本")
        st.write(df.head())
        st.divider()  # 添加分隔线

        # Section 2: 数据统计分析
        st.write("### 数据统计分析")
        
        # 显示 CAMPAIGN ID 的数量
        st.write("##### 基本统计信息")
        unique_campaigns = df['CAMPAIGN ID'].nunique()
        st.write(f"**CAMPAIGN ID 总数:** {unique_campaigns}")

        # 显示日期范围
        date_range = (df['DATE'].min(), df['DATE'].max())
        st.write(f"**时间范围:** {date_range[0]} 至 {date_range[1]}")

        # 各参数的最大、最小和平均值
        st.write("### 参数统计值")
        numeric_columns = ['TOTAL SALES', 'IMPRESSIONS', 'CLICKS', 'SPEND', 'CPC', 'CPM', 'CTR', 'ROAS']
        available_columns = [col for col in numeric_columns if col in df.columns]

        if available_columns:
            stats = df[available_columns].agg(['max', 'min', 'mean']).T
            stats.columns = ['最大值', '最小值', '平均值']
            stats.loc['CTR', '平均值'] = df['CTR'].mean() if 'CTR' in df.columns else 'N/A'
            stats.loc['ROAS', '平均值'] = df['ROAS'].mean() if 'ROAS' in df.columns else 'N/A'
            st.dataframe(stats.style.format("{:.2f}"))

        # Section 3: 数据分布图表
        st.write("## 数据分布图表")
        selected_param = st.selectbox("选择参数查看分布", options=available_columns, index=0)
        if selected_param:
            fig_hist = px.histogram(df, x=selected_param, title=f"{selected_param} 的分布", template='plotly_white')
            st.plotly_chart(fig_hist, use_container_width=True)

        # Section 4: 各 CAMPAIGN ID 的参数总览
        st.write("## 各 CAMPAIGN ID 的参数总览")
        aggregation = st.selectbox("选择统计字段", options=available_columns, index=0)
        if aggregation:
            # 聚合方式选择
            if aggregation in ['CTR', 'ROAS']:
                campaign_summary = df.groupby('CAMPAIGN ID')[aggregation].mean().reset_index()
            else:
                campaign_summary = df.groupby('CAMPAIGN ID')[aggregation].sum().reset_index()
            
            campaign_summary = campaign_summary.sort_values(by=aggregation, ascending=False)
            campaign_summary['CAMPAIGN ID'] = campaign_summary['CAMPAIGN ID'].astype(str)
            numeric_columns = campaign_summary.select_dtypes(include=['float64', 'int64']).columns
            numeric_columns = [col for col in numeric_columns if col != 'CAMPAIGN ID']
            campaign_summary[numeric_columns] = campaign_summary[numeric_columns].applymap(lambda x: f"{x:.2f}")
            st.dataframe(campaign_summary)

        # Section 5: 各 CAMPAIGN ID 的排名
        st.write("## 各 CAMPAIGN ID 的排名（平均值）")
        ranked_campaigns = df.groupby('CAMPAIGN ID')[available_columns].mean()
        ranked_campaigns = ranked_campaigns.sort_values(by=aggregation, ascending=False).reset_index()
        ranked_campaigns['CAMPAIGN ID'] = ranked_campaigns['CAMPAIGN ID'].astype(str)
        numeric_columns = ranked_campaigns.select_dtypes(include=['float64', 'int64']).columns
        ranked_campaigns[numeric_columns] = ranked_campaigns[numeric_columns].applymap(lambda x: f"{x:.2f}")
        st.dataframe(ranked_campaigns)





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
        # 新功能：用户选择单个CAMPAIGN ID并比较两个参数
        st.subheader("参数比较")
        selected_campaign_id = st.selectbox(
            "选择一个CAMPAIGN ID:",
            options=df['CAMPAIGN ID'].unique()
        )

        # 用户拖拽选择两个参数
        selected_params = st.multiselect(
            "选择两个参数进行比较:",
            options=available_columns,
            default=available_columns[:2]
        )

        if len(selected_params) == 2:
            # 数据按日期汇总
            comparison_data = (
                df[df['CAMPAIGN ID'] == selected_campaign_id]
                .groupby('DATE')[selected_params]
                .sum()
                .reset_index()
            )

            # 计算总值或平均值及排名
            total_summary = df.groupby('CAMPAIGN ID')[selected_params].agg(
                {param: ('mean' if param in ['CTR', 'ROAS'] else 'sum') for param in selected_params}
            ).rank(ascending=False, method='min').reset_index()

            campaign_totals = df.groupby('CAMPAIGN ID')[selected_params].agg(
                {param: 'mean' if param in ['CTR', 'ROAS'] else 'sum' for param in selected_params}
            ).reset_index()

            # 获取当前 Campaign 的总值或平均值和排名
            total_values = {}
            ranks = {}
            for param in selected_params:
                total_values[param] = campaign_totals.loc[
                    campaign_totals['CAMPAIGN ID'] == selected_campaign_id, param
                ].values[0]
                ranks[param] = int(total_summary.loc[
                    total_summary['CAMPAIGN ID'] == selected_campaign_id, param
                ])

            # 映射单位
            units = {param: units_mapping.get(param, '') for param in selected_params}

            # 创建双轴折线图
            fig_comparison = go.Figure()

            # 添加第一个参数
            fig_comparison.add_trace(
                go.Scatter(
                    x=comparison_data['DATE'],
                    y=comparison_data[selected_params[0]],
                    mode='lines+markers',
                    name=f"{selected_params[0]} (排名: {ranks[selected_params[0]]})",
                    yaxis="y1"
                )
            )

            # 添加第二个参数
            fig_comparison.add_trace(
                go.Scatter(
                    x=comparison_data['DATE'],
                    y=comparison_data[selected_params[1]],
                    mode='lines+markers',
                    name=f"{selected_params[1]} (排名: {ranks[selected_params[1]]})",
                    yaxis="y2"
                )
            )

            # 设置双轴
            fig_comparison.update_layout(
                title=f"Dual-Axis Comparison for CAMPAIGN ID: {selected_campaign_id}",
                xaxis=dict(title='DATE'),
                yaxis=dict(title=f"{selected_params[0]} ({units[selected_params[0]]})", side='left'),
                yaxis2=dict(title=f"{selected_params[1]} ({units[selected_params[1]]})", overlaying='y', side='right'),
                template='plotly_white',
                hovermode='x unified'
            )


            # 在 Streamlit 显示图表
            st.plotly_chart(fig_comparison, use_container_width=True)

            # 在 Streamlit 界面显示总数和排名
            st.write(f"### 周期内总值/平均值和排名")
            col1, col2 = st.columns(2)
            col1.metric(
                label=f"{selected_params[0]} 平均值/总值 (排名: {ranks[selected_params[0]]})",
                value=f"{total_values[selected_params[0]]:.2f} {units[selected_params[0]]}"
            )
            col2.metric(
                label=f"{selected_params[1]} 平均值/总值 (排名: {ranks[selected_params[1]]})",
                value=f"{total_values[selected_params[1]]:.2f} {units[selected_params[1]]}"
            )

        else:
            st.warning("请确保选择了两个参数进行比较！")



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
