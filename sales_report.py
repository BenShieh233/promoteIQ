import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plot_chart import *
from utils import *
from config import NUMERIC_COLUMNS, PAGES, UNITS_MAPPING
import math

# Streamlit 应用标题
st.set_page_config(page_title="广告趋势分析", layout="wide")
st.sidebar.title("广告趋势分析面板")
initialize_state()
selected_page = st.sidebar.selectbox("选择页面", PAGES)

# 数据导入页面
if selected_page == "数据导入":
    campaign_summary = upload_file("Upload your campaign summary", "campaign_summary")
    sales_report = upload_file("Upload your sales_report", "sales_report")
    # 进一步处理已上传的数据，先检查它们是否为 None
    if campaign_summary is not None and 'DATE' in campaign_summary.columns:
        campaign_summary['DATE'] = pd.to_datetime(campaign_summary['DATE']).dt.date
        campaign_summary['CAMPAIGN ID'] = campaign_summary['CAMPAIGN ID'].astype(str)
        st.session_state["uploaded_campaign_summary"] = campaign_summary
    if sales_report is not None and 'ORDER DATE' in sales_report.columns:
        sales_report['DATE'] = pd.to_datetime(sales_report['ORDER DATE']).dt.date  # 保留年月日
        sales_report['CAMPAIGN ID'] = sales_report['CAMPAIGN ID'].astype(str)
        st.session_state["uploaded_sales_report"] = sales_report

if selected_page == "数据预览":
    if not st.session_state["uploaded_campaign_summary"].empty:
        campaign_summary = st.session_state["uploaded_campaign_summary"]
        # 创建目录部分
        st.markdown("""
            <div style="position: sticky; top: 0; background: #f0f0f0; padding: 10px; border-radius: 5px;">
                <h3>目录</h3>
                <ul>
                    <li><a href="#section1">数据样本示例</a></li>
                    <li><a href="#section2">基本统计信息</a></li>
                    <li><a href="#section3">参数统计值</a></li>
                    <li><a href="#section4">CAMPAIGN ID 排名统计</a></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Section 1: 数据样本
        st.markdown('<a name="section1"></a>', unsafe_allow_html=True)
        st.write("##### 数据样本示例")
        st.write(campaign_summary.head())
        st.divider() 

        # Section 2: 数据统计分析
        # 显示 CAMPAIGN ID 的数量
        # 获取基本统计信息
        unique_campaigns = campaign_summary['CAMPAIGN ID'].nunique()
        date_range = (campaign_summary['DATE'].min(), campaign_summary['DATE'].max())

        # 使用Markdown表格显示
        st.markdown('<a name="section2"></a>', unsafe_allow_html=True)
        st.write("##### 基本统计信息")
        st.markdown(f"""
            <table style="width:100%; border-collapse: collapse;">
                <tr>
                    <td style="text-align: center; padding: 10px; border: 1px solid #ADD8E6; background-color: #f0f8ff;">
                        <strong style="color: #4682B4;">CAMPAIGN ID 总数</strong><br>
                        <span style="font-size: 24px; color: #2F4F4F; font-weight: bold;">{unique_campaigns}</span>
                    </td>
                    <td style="text-align: center; padding: 10px; border: 1px solid #ADD8E6; background-color: #f0f8ff;">
                        <strong style="color: #4682B4;">时间范围</strong><br>
                        <span style="font-size: 24px; color: #2F4F4F; font-weight: bold;">{date_range[0]} --- {date_range[1]}</span>
                    </td>
                </tr>
            </table>
        """, unsafe_allow_html=True)
        st.divider()

        # Section 3: 各参数的最大、最小和平均值
        st.markdown('<a name="section3"></a>', unsafe_allow_html=True)
        st.write("##### 参数统计值") 

        stats = campaign_summary[NUMERIC_COLUMNS].agg(['max', 'min', 'mean']).T
        stats.columns = ['最大值', '最小值', '平均值']
        stats.loc['CTR', '平均值'] = campaign_summary['CTR'].mean() if 'CTR' in campaign_summary.columns else 'N/A'
        stats.loc['ROAS', '平均值'] = campaign_summary['ROAS'].mean() if 'ROAS' in campaign_summary.columns else 'N/A'
        # 使用 Markdown 来居中整个表格
        html_table = stats.to_html(classes='table table-bordered', index=True)

        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                {html_table}
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.divider()

        # Section 4: 排名
        # 对数据按 'CAMPAIGN ID' 进行分组
        st.markdown('<a name="section4"></a>', unsafe_allow_html=True)
        grouped = campaign_summary.groupby('CAMPAIGN ID')[NUMERIC_COLUMNS].agg({
            'TOTAL SALES': 'sum',
            'IMPRESSIONS': 'sum',
            'CLICKS': 'sum',
            'SPEND': 'sum',
            'CPC': 'sum',
            'CPM': 'sum',
            'CTR': 'mean',
            'ROAS': 'mean'
        })
        grouped = grouped.round(2)
        st.write("##### 各 CAMPAIGN ID 参数排名概览")
        styled_ranked_campaigns = grouped.style.format(precision=2).background_gradient(cmap="coolwarm", subset=NUMERIC_COLUMNS)
        # 定义对齐样式
        centered_style = (
            grouped.style.format(precision=2).background_gradient(cmap="coolwarm", subset=NUMERIC_COLUMNS)
            .set_properties(**{'text-align': 'center'})  # 设置文字居中
            .set_table_styles([  # 设置表头和表格单元格的对齐方式
                {'selector': 'th', 'props': [('text-align', 'center')]},
                {'selector': 'td', 'props': [('text-align', 'center')]}
            ])
        )

        # 使用 Streamlit 展示
        st.dataframe(centered_style, use_container_width=True)        

if selected_page == "趋势分析":
    if not st.session_state["uploaded_campaign_summary"].empty:
        df = st.session_state["uploaded_campaign_summary"]
        st.subheader("趋势分析")
        aggregation_field = st.sidebar.selectbox("选择需要分析的参数:", options=NUMERIC_COLUMNS, index=0)
        date_range = st.sidebar.date_input(
                "选择需要分析的时间段:",
                value=[df['DATE'].min(), df['DATE'].max()],
                min_value=df['DATE'].min(),
                max_value=df['DATE'].max(),
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
        aggregated_summary, total_summary = process_data(data=df,
                                                         column_to_aggregate=aggregation_field,
                                                         min_rank=min_rank,
                                                         max_rank=max_rank,
                                                         date_range=date_range)


        fig_linechart, fig_data = plot_linechart(aggregated_summary,aggregation_field, total_summary)

        st.plotly_chart(fig_linechart, use_container_width=True)   

        st.subheader("参数比较")
        # 获取排名范围内的 CAMPAIGN ID
        campaign_ids = total_summary.iloc[min_rank - 1:max_rank].index

        # 提取每个 CAMPAIGN ID 的颜色
        campaign_colors = {trace.name: trace.line.color for trace in fig_data}

        # 创建按钮并处理用户点击，使用列来并排显示按钮
        st.write("### 选择一个 Campaign 进行分析：")

        # 分页参数
        items_per_page = 10
        total_pages = math.ceil(len(campaign_ids) / items_per_page)
        current_page = st.number_input('选择页面:', min_value=1, max_value=total_pages, value=1)

        # 获取当前页面要显示的按钮
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(campaign_ids))
        page_campaign_ids = campaign_ids[start_idx:end_idx]

        columns = st.columns(len(page_campaign_ids))  # 创建与当前页面按钮数量相等的列

        def on_campaign_button_click(campaign_id):
            st.session_state.selected_campaign_id = campaign_id

        for idx, campaign_id in enumerate(page_campaign_ids):
            with columns[idx]:
                if st.button(f"{campaign_id}", key=campaign_id, help=f"分析 {campaign_id}", use_container_width=True):
                    on_campaign_button_click(campaign_id)

        # 如果选定了 Campaign ID，则继续显示参数选择和图表
        # 如果有选定的 Campaign ID，显示该选择

        if 'selected_campaign_id' in st.session_state:
            selected_campaign_id = st.session_state.selected_campaign_id
            # 用户选择两个参数进行比较
            selected_params = st.multiselect("选择两个参数进行比较:", options=NUMERIC_COLUMNS, default=['TOTAL SALES', 'SPEND'])
            if len(selected_params) == 2:
                fig_comparison, total_values, ranks = create_comparison_chart(df, selected_campaign_id, selected_params, UNITS_MAPPING, date_range=date_range)
                # 显示图表
                st.plotly_chart(fig_comparison, use_container_width=True)

                # 显示总值和排名
                st.write(f"### 周期内总值/平均值和排名")
                col1, col2 = st.columns(2)
                col1.metric(
                    label=f"{selected_params[0]} 平均值/总值 (排名: {ranks[selected_params[0]]})",
                    value=f"{total_values[selected_params[0]]:.2f} {UNITS_MAPPING.get(selected_params[0], '')}"
                )
                col2.metric(
                    label=f"{selected_params[1]} 平均值/总值 (排名: {ranks[selected_params[1]]})",
                    value=f"{total_values[selected_params[1]]:.2f} {UNITS_MAPPING.get(selected_params[1], '')}"
                )

                st.divider()

                # 筛选逻辑
                df = df[(df['DATE'] >= date_range[0]) & (df['DATE'] <= date_range[1])]
                filtered_df = df[df['CAMPAIGN ID'] == selected_campaign_id]

                # 提取所需列，并去重
                if not filtered_df.empty:
                    # 转换为整数并移除 .0
                    filtered_df['PROMOTED PRODUCT / CREATIVE ID'] = (
                        filtered_df['PROMOTED PRODUCT / CREATIVE ID']
                        .fillna(0)  # 避免 NaN 引发问题
                        .apply(lambda x: str(int(x)) if pd.notnull(x) else "")
                    )
                    # 转换为整数并移除 .0
                    filtered_df['PROMOTED SKU'] = (
                        filtered_df['PROMOTED SKU']
                        .fillna(0)  # 避免 NaN 引发问题
                        .apply(lambda x: str(int(x)) if pd.notnull(x) else "")
                    )
                    unique_products = filtered_df[['PROMOTED SKU', 'PROMOTED PRODUCT / CREATIVE ID', 'PROMOTED PRODUCT / CREATIVE']].drop_duplicates().reset_index(drop=True)
                    st.write(f"##### Campaign ID {selected_campaign_id} - 包含的所有Promoted SKU")
                    st.dataframe(unique_products)
                else:
                    st.write("No data available for the selected Campaign ID.")

                # 绘图
                # 按不同方式聚合数据
                aggregated_df = (
                    filtered_df.groupby(['DATE', 'PROMOTED SKU'], as_index=False)
                    .agg({
                        'TOTAL SALES': 'sum',
                        'SPEND': 'sum',
                        'IMPRESSIONS': 'sum',
                        'CLICKS': 'sum',
                        'CPC': 'sum',
                        'CPM': 'sum',
                        'CTR': 'mean',
                        'ROAS': 'mean'
                    })
                )
                if not aggregated_df.empty:
                    all_metrics = NUMERIC_COLUMNS

                    # 使用 st.columns 将两个下拉框放在同一行
                    col1, col2 = st.columns(2)

                    with col1:
                        metric1 = st.selectbox("选择第一个指标", options=all_metrics, index=0)  # 默认选择第一个指标
                    with col2:
                        metric2 = st.selectbox("选择第二个指标", options=all_metrics, index=1)  # 默认选择第二个指标

                    # 检查用户是否选择了相同的参数
                    if metric1 == metric2:
                        st.error("两个指标不能相同，请选择不同的指标。")
                    else:
                        # 绘制第一个指标图
                        st.write(f"### Promoted SKU 在 {metric1} 中的趋势")
                        fig1 = px.line(
                            aggregated_df,
                            x="DATE",
                            y=metric1,
                            color="PROMOTED SKU",
                            markers=True,
                            title=f"{metric1} Trend",
                            labels={"DATE": "Date", metric1: metric1, "PROMOTED SKU": "SKU"},
                        )
                        fig1.update_layout(legend_title="Promoted SKU", xaxis_title="Date", yaxis_title=metric1)
                        st.plotly_chart(fig1)

                        st.write(f"### Promoted SKU 在 {metric2} 中的趋势")
                        fig2 = px.line(
                            aggregated_df,
                            x="DATE",
                            y=metric2,
                            color="PROMOTED SKU",
                            markers=True,
                            title=f"{metric2} Trend",
                            labels={"DATE": "Date", metric2: metric2, "PROMOTED SKU": "SKU"},
                        )
                        fig2.update_layout(legend_title="Promoted SKU", xaxis_title="Date", yaxis_title=metric2)
                        st.plotly_chart(fig2)
                else:
                    st.write("No data available for the selected SKUs.")
            else:
                st.warning("请确保选择了两个参数进行比较！")                
        else:
            st.warning("请选择一个 Campaign ID！")

if selected_page == "百分比分布":
    if not st.session_state["uploaded_campaign_summary"].empty:
        df = st.session_state["uploaded_campaign_summary"]

        def generate_distribution_analysis(df, groupby_field):
            """
            Function to generate percentage distribution and plotting for a given groupby field.

            Parameters:
                df (DataFrame): The uploaded campaign summary data.
                groupby_field (str): The field to group by ('CAMPAIGN ID' or 'PROMOTED SKU').
            """
            st.subheader(f"百分比分布分析 - 按 {groupby_field}")
            col1, col2 = st.columns(2)
            
            with col1:
                aggregation_field = st.selectbox("选择需要分析的参数:", options=NUMERIC_COLUMNS, index=0)
            with col2:
                date_range = st.date_input(
                    "选择需要分析的时间段:",
                    value=[df['DATE'].min(), df['DATE'].max()],
                    min_value=df['DATE'].min(),
                    max_value=df['DATE'].max(),
                )
            
            combine_others = st.checkbox("将其余分组归类为 Others", value=True)
            filtered_df = df[(df['DATE'] >= date_range[0]) & (df['DATE'] <= date_range[1])]

            agg_func = calculate_by_column_type(aggregation_field)
            df_summary = filtered_df.groupby(groupby_field).agg({aggregation_field: agg_func}).sort_values(by=aggregation_field, ascending=False)

            top_n = st.slider(
                f"选择展示前 N 个{groupby_field}:",
                min_value=1,
                max_value=len(df_summary),
                value=min(5, len(df_summary)),
            )

            top_groups = df_summary.head(top_n).index.tolist()
            other_groups = df_summary[~df_summary.index.isin(top_groups)].index.tolist()

            sum_columns = ['TOTAL SALES', 'IMPRESSIONS', 'CLICKS', 'SPEND', 'CPC', 'CPM']
            mean_columns = ['CTR', 'ROAS']

            if aggregation_field in sum_columns:
                for i in range(0, len(sum_columns), 2):
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_piechart(filtered_df, groupby_field, top_groups, other_groups, sum_columns[i], combine_others)
                        st.plotly_chart(fig, key=f"{groupby_field}_pie_{i}")

                    if i + 1 < len(sum_columns):
                        with col2:
                            fig = plot_piechart(filtered_df, groupby_field, top_groups, other_groups, sum_columns[i + 1], combine_others)
                            st.plotly_chart(fig, key=f"{groupby_field}_pie_{i + 1}")

                for i in range(0, len(mean_columns), 2):
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_percentage_histogram(filtered_df, groupby_field, top_groups, mean_columns[i])
                        st.plotly_chart(fig, key=f"{groupby_field}_hist_{mean_columns[i]}_1")

                    if i + 1 < len(mean_columns):
                        with col2:
                            fig = plot_percentage_histogram(filtered_df, groupby_field, top_groups, mean_columns[i + 1])
                            st.plotly_chart(fig, key=f"{groupby_field}_hist_{mean_columns[i + 1]}_2")

            else:
                for i in range(0, len(mean_columns), 2):
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_percentage_histogram(filtered_df, groupby_field, top_groups, mean_columns[i])
                        st.plotly_chart(fig, key=f"{groupby_field}_hist_{mean_columns[i]}_1")

                    if i + 1 < len(mean_columns):
                        with col2:
                            fig = plot_percentage_histogram(filtered_df, groupby_field, top_groups, mean_columns[i + 1])
                            st.plotly_chart(fig, key=f"{groupby_field}_hist_{mean_columns[i + 1]}_2")

                for i in range(0, len(sum_columns), 2):
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_piechart(filtered_df, groupby_field, top_groups, other_groups, sum_columns[i], combine_others)
                        st.plotly_chart(fig, key=f"{groupby_field}_pie_{i}")

                    if i + 1 < len(sum_columns):
                        with col2:
                            fig = plot_piechart(filtered_df, groupby_field, top_groups, other_groups, sum_columns[i + 1], combine_others)
                            st.plotly_chart(fig, key=f"{groupby_field}_pie_{i + 1}")
        # 为用户提供选择分组字段的选项
        groupby_field = st.sidebar.radio(
            "选择分组字段:",
            options=['CAMPAIGN ID', 'PROMOTED SKU'],  # 提供选项
            index=0  # 默认选择第一个选项
        )

        # 调用分组分析函数
        generate_distribution_analysis(df, groupby_field)

    else:
        st.warning("请检查上传文件格式是否正确")

if selected_page == "关联销售额分析":
    try:
        if not st.session_state["uploaded_campaign_summary"].empty:
            campaign = st.session_state["uploaded_campaign_summary"]
            if not st.session_state["uploaded_sales_report"].empty:
                sales = st.session_state["uploaded_sales_report"]
            # 用户选择日期范围

            date_range = st.sidebar.date_input(
                "选择需要分析的时间段:",
                value=[sales['DATE'].min(), sales['DATE'].max()],
                min_value=sales['DATE'].min(),
                max_value=sales['DATE'].max(),
            )

            sales = sales[(sales['DATE'] >= date_range[0]) & (sales['DATE'] <= date_range[1])]

            st.subheader("关联销售额分析")

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
            selected_campaign_id = st.sidebar.selectbox(
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

            st.write(filtered_sales)
        else:
            st.warning("请检查是否同时上传了campaign summary和sales report两份Excel文件")
    except Exception:
        st.warning("请检查是否同时上传了 Campaign Performance 和 sales Report 两份文件")

if selected_page == "广告活动与SKU各指标流向":
    if not st.session_state["uploaded_campaign_summary"].empty:
        df = st.session_state["uploaded_campaign_summary"]
                # 去除包含 NaN 值的行
        df_cleaned = df.dropna(subset=['CAMPAIGN ID', 'PROMOTED SKU'])

        # 将 CAMPAIGN ID 和 PROMOTED SKU 转换为不带小数点的字符串
        df_cleaned['CAMPAIGN ID'] = df_cleaned['CAMPAIGN ID'].astype(str).apply(lambda x: x.split('.')[0])
        df_cleaned['PROMOTED SKU'] = df_cleaned['PROMOTED SKU'].astype(str).apply(lambda x: x.split('.')[0])
        df_cleaned['DATE'] = pd.to_datetime(df_cleaned['DATE'], errors='coerce')


        st.subheader("广告活动与SKU各指标流向")
        primary_choice = st.sidebar.radio("选择初始关系图展示方式", ["CAMPAIGN ID", "PROMOTED SKU"])

        if primary_choice == "CAMPAIGN ID":
            unique_primary_values = df_cleaned['CAMPAIGN ID'].unique()
            primary_label = 'CAMPAIGN ID'
            secondary_label = 'PROMOTED SKU'
        else:
            unique_primary_values = df_cleaned['PROMOTED SKU'].unique()
            primary_label = 'PROMOTED SKU'
            secondary_label = 'CAMPAIGN ID'

        metrics = NUMERIC_COLUMNS

        if 'selected_campaign_id' in st.session_state:
            selected_campaign_id = st.session_state.selected_campaign_id
        else:
            selected_campaign_id = None  # 没有选择时设置为 None

        # 设置默认值
        if selected_campaign_id:
            # 如果已选定 Campaign ID, 设置为该值
            default_primary = selected_campaign_id if selected_campaign_id in unique_primary_values else unique_primary_values[0]
        else:
            # 如果没有选定 Campaign ID，设置第一个唯一值为默认值
            default_primary = unique_primary_values[0]


        if selected_campaign_id:
            selected_primary = st.sidebar.selectbox(f"选择一个 {primary_label}", unique_primary_values, index=list(unique_primary_values).index(default_primary))
        else:
            selected_primary = st.sidebar.selectbox(f"选择一个 {primary_label}", unique_primary_values)    

        selected_metric = st.sidebar.selectbox("选择指标", metrics, index=metrics.index('TOTAL SALES'))

        date_range = st.sidebar.date_input("选择日期范围", [df_cleaned['DATE'].min(), df_cleaned['DATE'].max()])

        grouped_df, fig = plot_sankey(df_cleaned, 
                        selected_primary=selected_primary,
                        selected_metric=selected_metric,
                        date_range=date_range,
                        primary_label=primary_label,
                        secondary_label=secondary_label)
        
        st.plotly_chart(fig)
        st.write(grouped_df)
    else:
        st.warning("请检查文件格式是否正确")

if selected_page == "广告效果评分":
    if not st.session_state["uploaded_campaign_summary"].empty:
        df = st.session_state["uploaded_campaign_summary"]

        # 日期范围选择
        st.sidebar.subheader("选择统计周期")
        min_date, max_date = df["DATE"].min(), df["DATE"].max()
        date_range = st.sidebar.date_input(
            "选择日期范围:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
        )
        # 如果用户未选择结束日期，自动设置为最大日期
        if len(date_range) == 1:
            start_date = date_range[0]
            end_date = max_date
        else:
            start_date, end_date = date_range
        df = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

        # Streamlit Sidebar
        st.sidebar.header("选择分析维度和权重")

        # 选择统计轴
        axis_option = st.sidebar.selectbox("统计轴", options=["CAMPAIGN ID", "PROMOTED SKU"])

        # 使用 slider 设置权重为整数
        w1 = st.sidebar.slider("TOTAL SALES 权重 (w1)", 0, 100, 33, step=1)
        w2 = st.sidebar.slider("ROAS 权重 (w2)", 0, 100, 33, step=1)
        w3 = st.sidebar.slider("SPEND 权重 (w3)", 0, 100, step=1)

        # 确保总权重为 100%
        total_weight = w1 + w2 + w3
        if total_weight != 100:
            st.sidebar.warning(f"权重总和必须为 100。当前总和为 {total_weight}")
        else:
            # 将权重转换为比例
            w1 /= 100
            w2 /= 100
            w3 /= 100

            # 数据聚合
            grouped = df.groupby(axis_option).agg(
                TOTAL_SALES=("TOTAL SALES", "sum"),
                SPEND=("SPEND", "sum"),
                ROAS=("ROAS", "mean"),
            ).reset_index()

            # 数据归一化
            grouped["TOTAL_SALES_NORM"] = (grouped["TOTAL_SALES"] - grouped["TOTAL_SALES"].min()) / \
                                        (grouped["TOTAL_SALES"].max() - grouped["TOTAL_SALES"].min())
            grouped["ROAS_NORM"] = (grouped["ROAS"] - grouped["ROAS"].min()) / \
                                (grouped["ROAS"].max() - grouped["ROAS"].min())
            grouped["SPEND_NORM"] = (grouped["SPEND"] - grouped["SPEND"].min()) / \
                                    (grouped["SPEND"].max() - grouped["SPEND"].min())

            # 百分比加权评分
            grouped["Score"] = (
                w1 * grouped["TOTAL_SALES_NORM"] +
                w2 * grouped["ROAS_NORM"] -
                w3 * grouped["SPEND_NORM"]
            )

            # 动态显示结果
            st.write(f"按 {axis_option} 汇总的结果")
            if axis_option == "CAMPAIGN ID":
                st.dataframe(grouped[["CAMPAIGN ID", "TOTAL_SALES", "SPEND", "ROAS", "Score"]])
            elif axis_option == "PROMOTED SKU":
                st.dataframe(grouped[["PROMOTED SKU", "TOTAL_SALES", "SPEND", "ROAS", "Score"]])

if selected_page == "PLA vs Banner":
    if not st.session_state["uploaded_campaign_summary"].empty:
        df = st.session_state["uploaded_campaign_summary"]

        # 日期范围选择
        st.sidebar.subheader("选择统计周期")
        min_date, max_date = df["DATE"].min(), df["DATE"].max()
        date_range = st.sidebar.date_input(
            "选择日期范围:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
        )
        # 如果用户未选择结束日期，自动设置为最大日期
        if len(date_range) == 1:
            start_date = date_range[0]
            end_date = max_date
        else:
            start_date, end_date = date_range

        df = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

        chart_type = st.sidebar.radio(
            "选择图表类型：",
            options=["百分比饼图", "折线图"]
        )
        # 聚合操作
        aggregations = {
            'IMPRESSIONS': 'sum',
            'CLICKS': 'sum',
            'SPEND': 'sum',
            'CPC': 'sum',
            'CPM': 'sum',
            'CTR': 'mean',
            'TOTAL SALES': 'sum',
            'ROAS': 'mean',
        }

        # 百分比饼图逻辑
        if chart_type == "百分比饼图":
            # 按 PLACEMENT TYPE 聚合
            grouped = df.groupby('PLACEMENT TYPE').agg(aggregations).reset_index()
            total = grouped.iloc[:, 1:].sum(axis=0)
            percent_df = grouped.copy()
            for col in grouped.columns[1:]:
                percent_df[col] = grouped[col] / total[col] * 100

            st.title("广告类型百分比饼图")
            columns = list(grouped.columns[1:])

            # 每行展示两张图
            for i in range(0, len(columns), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(columns):
                        col_name = columns[i + j]
                        fig = px.pie(
                            percent_df,
                            names='PLACEMENT TYPE',
                            values=col_name,
                            title=f'{col_name} 百分比分布',
                            hole=0.4
                        )
                        cols[j].plotly_chart(fig, use_container_width=True)

        # 折线图逻辑
        elif chart_type == "折线图":
            # 按 DATE 和 PLACEMENT TYPE 聚合
            date_grouped = df.groupby(['DATE', 'PLACEMENT TYPE']).agg(aggregations).reset_index()

            st.title("参数时间折线图")
            columns = list(aggregations.keys())

            # 每行展示两张图
            for i in range(0, len(columns), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(columns):
                        col_name = columns[i + j]
                        fig = px.line(
                            date_grouped,
                            x='DATE',
                            y=col_name,
                            color='PLACEMENT TYPE',
                            title=f'{col_name} 日期趋势',
                            markers=True
                        )
                        cols[j].plotly_chart(fig, use_container_width=True)