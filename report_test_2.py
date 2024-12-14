import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plot_chart import plot_piechart, plot_percentage_histogram
from utils import *

# Streamlit 应用标题
st.set_page_config(page_title="广告趋势分析", layout="wide")
st.title("广告趋势分析面板")

# 初始化 Session State
if "uploaded_campaign_summary" not in st.session_state:
    st.session_state["uploaded_campaign_summary"] = pd.DataFrame()
if "uploaded_sales_report" not in st.session_state:
    st.session_state["uploaded_sales_report"] = pd.DataFrame()

# 页面选项
pages = ["数据导入", "数据预览", "趋势分析", "百分比分布", "关联销售额分析"]
selected_page = st.sidebar.selectbox("选择页面", pages)
numeric_columns = ['TOTAL SALES', 'IMPRESSIONS', 'CLICKS', 'SPEND', 'CPC', 'CPM', 'CTR', 'ROAS']
available_columns = [col for col in numeric_columns if col in st.session_state["uploaded_campaign_summary"].columns]
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

# 数据导入页面
if selected_page == "数据导入":
    campaign_summary = upload_file("Upload your campaign summary", "campaign_summary")
    sales_report = upload_file("Upload your sales_report", "sales_report")
    if not campaign_summary.empty:
        st.session_state["uploaded_campaign_summary"] = campaign_summary
        if 'DATE' in campaign_summary.columns:
            campaign_summary['DATE'] = pd.to_datetime(campaign_summary['DATE']).dt.date    
        else:
            st.warning("请上传包含有效数据的文件进行分析")
    if not sales_report.empty:
        st.session_state["uploaded_sales_report"] = sales_report
        if 'ORDER DATE' in sales_report.columns:
            sales_report['DATE'] = pd.to_datetime(sales_report['ORDER DATE']).dt.date  # 保留年月日
        else:
            st.warning("请上传包含有效数据的文件进行分析")
    else:
        st.info("请上传格式为.xlsx的Excel文件")

# 数据预览页面
if selected_page == "数据预览":
    if not st.session_state["uploaded_campaign_summary"].empty:
        df = st.session_state["uploaded_campaign_summary"]
        # Section 1: 数据样本
        st.write(df.head())
        st.divider() 

        # Section 2: 数据统计分析
        st.write("### 数据统计分析")
        
        # 显示 CAMPAIGN ID 的数量
        st.write("##### 基本统计信息")
        unique_campaigns = df['CAMPAIGN ID'].nunique()
        st.write(f"**CAMPAIGN ID 总数:** {unique_campaigns}")

        # 显示日期范围
        date_range = (df['DATE'].min(), df['DATE'].max())
        st.write(f"**时间范围:** {date_range[0]} 至 {date_range[1]}")
        st.divider()  # 添加分隔线  

        # 各参数的最大、最小和平均值
        st.write("##### 参数统计值")
        numeric_columns = ['TOTAL SALES', 'IMPRESSIONS', 'CLICKS', 'SPEND', 'CPC', 'CPM', 'CTR', 'ROAS']
        available_columns = [col for col in numeric_columns if col in df.columns]

        if available_columns:
            stats = df[available_columns].agg(['max', 'min', 'mean']).T
            stats.columns = ['最大值', '最小值', '平均值']
            stats.loc['CTR', '平均值'] = df['CTR'].mean() if 'CTR' in df.columns else 'N/A'
            stats.loc['ROAS', '平均值'] = df['ROAS'].mean() if 'ROAS' in df.columns else 'N/A'
            st.dataframe(stats.style.format("{:.2f}"))

        st.divider()  # 添加分隔线

        # Section 3: 数据分布图表
        st.write("### 数据分布图表")
        selected_param = st.selectbox("选择参数查看分布", options=available_columns, index=0)
        if selected_param:
            fig_hist = px.histogram(df, x=selected_param, title=f"{selected_param} 的分布", template='plotly_white')
            st.plotly_chart(fig_hist, use_container_width=True)

        # Section 4: 各 CAMPAIGN ID 的参数总览
        st.write("##### 各 CAMPAIGN ID 的参数总览")
        aggregation = st.selectbox("选择统计字段", options=available_columns, index=0)

        st.divider()  # 添加分隔线

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
            campaign_summary[numeric_columns] = campaign_summary[numeric_columns].map(lambda x: f"{x:.2f}")
            st.dataframe(campaign_summary)

        st.divider()  # 添加分隔线

        # Section 5: 各 CAMPAIGN ID 的排名
        ranked_campaigns = df.groupby('CAMPAIGN ID')[available_columns].mean()
        ranked_campaigns = ranked_campaigns.sort_values(by=aggregation, ascending=False).reset_index()
        ranked_campaigns['CAMPAIGN ID'] = ranked_campaigns['CAMPAIGN ID'].astype(str)
        numeric_columns = ranked_campaigns.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            ranked_campaigns[col] = pd.to_numeric(ranked_campaigns[col], errors='coerce')
        ranked_campaigns[numeric_columns] = ranked_campaigns[numeric_columns].map(lambda x: round(x, 2) if pd.notna(x) else x)
        ranked_campaigns[numeric_columns] = ranked_campaigns[numeric_columns].round(2)  # 修改底层数据为两位小数

        # 应用样式
        st.write("##### 各 CAMPAIGN ID 参数平均值概览")
        styled_ranked_campaigns = ranked_campaigns.style.format(precision=2).background_gradient(cmap="coolwarm", subset=numeric_columns)
        st.dataframe(styled_ranked_campaigns)

if selected_page == "趋势分析" and available_columns:
    if not st.session_state["uploaded_campaign_summary"].empty:
        df = st.session_state["uploaded_campaign_summary"]
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

if selected_page == "百分比分布" and available_columns:
    if not st.session_state["uploaded_campaign_summary"].empty:
        df = st.session_state["uploaded_campaign_summary"]

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
        top_campaigns = top_n_campaigns.index.tolist()
        other_campaigns = specific_date_summary[~specific_date_summary.isin(top_campaigns)].index.tolist()
        

        combine_others = st.checkbox("将其余 Campaign ID 归类为 Others", value=True)

        # 计算图表数量
        num_columns = len(numeric_columns)
        fig_aggregation_field = plot_piechart(specific_date_data,
                                                    "CAMPAIGN ID",
                                                    str(selected_date),
                                                    top_campaigns,
                                                    other_campaigns,
                                                    aggregation_field,
                                                    combine_others)

        # 使用 st.columns 创建布局
        # 第一行必须有两个图表
        cols = st.columns(2)
        with cols[0]:
            st.plotly_chart(fig_aggregation_field)  # 用户指定的图表

        # 第一行第二个位置依次填充剩余的图表
        with cols[1]:
            if len(numeric_columns) > 1:  # 检查是否有其他图表
                first_other_column = numeric_columns[1]
                if first_other_column in ['CTR', 'ROAS']:
                    fig = plot_percentage_histogram(
                        data=specific_date_data,
                        column_to_aggregate='CAMPAIGN ID',
                        selected_date=str(selected_date),
                        top_campaigns=top_campaigns,
                        aggregation_column=first_other_column
                    )
                else:
                    fig = plot_piechart(
                        data=specific_date_data, 
                        column_to_aggregate='CAMPAIGN ID',
                        selected_date=str(selected_date), 
                        top_campaigns=top_campaigns, 
                        other_campaigns=other_campaigns, 
                        aggregation_column=first_other_column,
                        combine_others=combine_others
                    )
                st.plotly_chart(fig)
            else:
                st.write("")  # 空白占位图表

        # 计算剩余的列
        remaining_columns = [col for col in numeric_columns if col != aggregation_field and col != first_other_column]
        num_remaining_columns = len(remaining_columns)

        # 创建后续图表布局，每行两个图表
        for i in range(0, num_remaining_columns, 2):
            cols = st.columns(2)
            with cols[0]:
                if remaining_columns[i] in ['CTR', 'ROAS']:
                    fig = plot_percentage_histogram(
                        data=specific_date_data, 
                        column_to_aggregate='CAMPAIGN ID',
                        selected_date=str(selected_date), 
                        top_campaigns=top_campaigns, 
                        aggregation_column=remaining_columns[i]
                    )
                else:
                    fig = plot_piechart(
                        data=specific_date_data, 
                        column_to_aggregate='CAMPAIGN ID',
                        selected_date=str(selected_date), 
                        top_campaigns=top_campaigns, 
                        other_campaigns=other_campaigns, 
                        aggregation_column=remaining_columns[i],
                        combine_others=combine_others
                    )
                st.plotly_chart(fig)
            if i + 1 < num_remaining_columns:  # 如果第二个图表存在
                with cols[1]:
                    if remaining_columns[i+1] in ['CTR', 'ROAS']:
                        fig = plot_percentage_histogram(
                            data=specific_date_data, 
                            column_to_aggregate='CAMPAIGN ID',
                            selected_date=str(selected_date), 
                            top_campaigns=top_campaigns, 
                            aggregation_column=remaining_columns[i+1]
                        )
                    else:
                        fig = plot_piechart(
                            data=specific_date_data, 
                            column_to_aggregate='CAMPAIGN ID',
                            selected_date=str(selected_date), 
                            top_campaigns=top_campaigns, 
                            other_campaigns=other_campaigns, 
                            aggregation_column=remaining_columns[i+1],
                            combine_others = combine_others
                        )
                    st.plotly_chart(fig)
            else:  # 如果没有第二个图表，空白占位
                with cols[1]:
                    st.write("")  # 空白占位图表

        # 如果最后剩下一个图表，单独占一行并居中
        if num_remaining_columns % 2 != 0:
            cols = st.columns(1)
            with cols[0]:
                if remaining_columns[-1] in ['CTR', 'ROAS']:
                    fig = plot_percentage_histogram(
                        data=specific_date_data, 
                        column_to_aggregate='CAMPAIGN ID',
                        selected_date=str(selected_date), 
                        top_campaigns=top_campaigns, 
                        aggregation_column=remaining_columns[-1]
                    )
                else:
                    fig = plot_piechart(
                        data=specific_date_data, 
                        column_to_aggregate='CAMPAIGN ID',
                        selected_date=str(selected_date), 
                        top_campaigns=top_campaigns, 
                        other_campaigns=other_campaigns, 
                        aggregation_column=remaining_columns[-1],
                        combine_others = combine_others                        
                    )
                st.plotly_chart(fig)
    else:
        st.warning("请检查上传文件格式是否正确")

if selected_page == "关联销售额分析":
    if not st.session_state["uploaded_campaign_summary"].empty or st.session_state["uploaded_sales_report"].empty:
        campaign = st.session_state["uploaded_campaign_summary"]
        sales = st.session_state["uploaded_sales_report"]
    else:
        st.warning("请检查是否同时上传了campaign summary和sales report两份Excel文件")

    st.subheader("关联销售额分析")
    st.write(sales)

    # 数据处理部分
    promoted_sku_aggregated = campaign.groupby(['CAMPAIGN ID', 'DATE'])['PROMOTED SKU'] \
        .apply(set).reset_index()

    merged = sales.merge(promoted_sku_aggregated, on=['CAMPAIGN ID', 'DATE'], how='left')
    merged['Is Promoted'] = merged.apply(lambda row: row['PURCHASED SKU'] in row['PROMOTED SKU'], axis=1)

    # 数据处理部分
    grouped_sales = merged.groupby(['CAMPAIGN ID', 'Is Promoted'])['TOTAL SALES'].sum().reset_index()
    grouped_sales['Category'] = grouped_sales['Is Promoted'].map({True: 'Promoted SKU', False: 'Non-Promoted SKU'})
    grouped_sales['Total'] = grouped_sales.groupby('CAMPAIGN ID')['TOTAL SALES'].transform('sum')
    grouped_sales['Sales Contribution (%)'] = (grouped_sales['TOTAL SALES'] / grouped_sales['Total']) * 100

    # 自定义配色方案，只为 SKU 层（Category）指定颜色
    color_map = {
        'Promoted SKU': 'blue',  
        'Non-Promoted SKU': 'pink'  # 只为 SKU 分类设定颜色
    }
    st.write(grouped_sales)
    # Sunburst 图
    fig = px.sunburst(
        grouped_sales,
        path=['CAMPAIGN ID', 'Category'],  # CAMPAIGN ID 只是路径，不参与颜色映射
        values='TOTAL SALES',
        color='Category',  # 只有 'Category'（SKU）层级有颜色
        color_discrete_map=color_map,  # 使用自定义配色方案
        title= "按活动 ID 分类的销售贡献（promoted SKU 与 non promoted SKU）"
    )

    fig.update_traces(textinfo="label+percent entry")

    # 更新布局，设置背景颜色等
    fig.update_layout(
        plot_bgcolor='white',  # 背景色为白色
        paper_bgcolor='white',  # 图表纸张背景色为白色
        title="按Campaign ID 分类的Total Sales贡献（promoted SKU 与 non promoted SKU）",
        height=800,
        width=1200,
        margin=dict(t=50, b=50, l=50, r=50)  # 调整边距
    )

    # 显示图表
    st.plotly_chart(fig, use_container_width=True)

    # 添加 CAMPAIGN ID 选择框，确保按升序排列
    selected_campaign_id = st.selectbox(
        "选择一个 CAMPAIGN ID 查看每日销售趋势:",
        options=sorted(merged['CAMPAIGN ID'].unique())  # 使用 sorted() 确保按升序排列
    )


    # 筛选所选 CAMPAIGN ID 的数据
    filtered_sales = merged[merged['CAMPAIGN ID'] == selected_campaign_id]
    daily_sales = filtered_sales.groupby(['DATE', 'Is Promoted'])['TOTAL SALES'].sum().reset_index()
    daily_sales['Category'] = daily_sales['Is Promoted'].map({True: 'Promoted SKU', False: 'Non-Promoted SKU'})

    # Plotly 柱状图：每日销售趋势
    fig2 = px.bar(
        daily_sales,
        x='DATE',
        y='TOTAL SALES',
        color='Category',
        barmode='group',
        title=f'Daily Sales Trend for {selected_campaign_id} (Promoted vs Non-Promoted SKUs)',
        labels={'TOTAL SALES': 'Sales Amount', 'DATE': 'Date'},
        color_discrete_map={'Promoted SKU': 'blue', 'Non-Promoted SKU': 'orange'}
    )
    fig2.update_xaxes(type='category')
    st.plotly_chart(fig2, use_container_width=True)