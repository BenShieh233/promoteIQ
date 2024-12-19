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
st.title("广告趋势分析面板")
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
        aggregation_field = st.selectbox("选择需要分析的参数:", options=NUMERIC_COLUMNS, index=0)
        # 用户选择日期范围
        date_range = st.date_input(
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
        items_per_page = 5
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
            selected_params = st.multiselect("选择两个参数进行比较:", options=NUMERIC_COLUMNS, default=NUMERIC_COLUMNS[:2])
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

            else:
                st.warning("请确保选择了两个参数进行比较！")
        else:
            st.warning("请选择一个 Campaign ID！")

if selected_page == "百分比分布":
    if not st.session_state["uploaded_campaign_summary"].empty:
        df = st.session_state["uploaded_campaign_summary"]

        st.subheader("百分比分布分析")
        aggregation_field = st.selectbox("选择需要分析的参数:", options=NUMERIC_COLUMNS)

        # 用户选择日期范围
        date_range = st.date_input(
            "选择需要分析的时间段:",
            value=[df['DATE'].min(), df['DATE'].max()],
            min_value=df['DATE'].min(),
            max_value=df['DATE'].max(),
        )

        combine_others = st.checkbox("将其余 Campaign ID 归类为 Others", value=True)

        filtered_df = df[(df['DATE'] >= date_range[0]) & (df['DATE'] <= date_range[1])]

        agg_func = calculate_by_column_type(aggregation_field)
        df_summary = filtered_df.groupby('CAMPAIGN ID').agg({aggregation_field: agg_func}).sort_values(by=aggregation_field, ascending=False)

        top_n = st.slider(
            "选择展示前 N 个Campaign ID:",
            min_value=1,
            max_value=len(df_summary),
            value=min(5, len(df_summary)),
        )
        top_n_campaigns = df_summary.head(top_n)

        top_campaigns = top_n_campaigns.index.tolist()
        other_campaigns = df_summary[~df_summary.isin(top_campaigns)].index.tolist()
        sum_columns = ['TOTAL SALES', 'IMPRESSIONS', 'CLICKS', 'SPEND', 'CPC', 'CPM']
        mean_columns = ['CTR', 'ROAS']

        if aggregation_field in sum_columns:
            # 排列顺序
            remaining_columns = [col for col in sum_columns if col != aggregation_field]
            plot_columns = [aggregation_field] + remaining_columns
            
            # 求和列显示：每行两张图
            for i in range(0, len(plot_columns), 2):
                col1, col2 = st.columns(2)  # 创建两列布局
                with col1:
                    fig_aggregation_field = plot_piechart(filtered_df,
                                                        "CAMPAIGN ID",
                                                        top_campaigns,
                                                        other_campaigns,
                                                        plot_columns[i],
                                                        combine_others
                    )
                    # 为每个图表设置唯一的 key
                    st.plotly_chart(fig_aggregation_field, key=f"piechart_{i}")
                
                if i + 1 < len(plot_columns):
                    with col2:
                        fig_aggregation_field = plot_piechart(filtered_df,
                                                            "CAMPAIGN ID",
                                                            top_campaigns,
                                                            other_campaigns,
                                                            plot_columns[i + 1],
                                                            combine_others
                        )
                        # 为每个图表设置唯一的 key
                        st.plotly_chart(fig_aggregation_field, key=f"piechart_{i + 1}")
            
            # 紧接着绘制均值列的直方图（如 CTR, ROAS）
            for i in range(0, len(mean_columns), 2):
                col1, col2 = st.columns(2)  # 每行两个图
                with col1:
                    fig_percentage_histogram = plot_percentage_histogram(filtered_df,
                                                                        "CAMPAIGN ID",
                                                                        top_campaigns,
                                                                        mean_columns[i]
                    )
                    # 为每个图表设置唯一的 key
                    st.plotly_chart(fig_percentage_histogram, key=f"histogram_{mean_columns[i]}_1")
                
                if i + 1 < len(mean_columns):
                    with col2:
                        fig_percentage_histogram = plot_percentage_histogram(filtered_df,
                                                                            "CAMPAIGN ID",
                                                                            top_campaigns,
                                                                            mean_columns[i + 1]
                        )
                        # 为每个图表设置唯一的 key
                        st.plotly_chart(fig_percentage_histogram, key=f"histogram_{mean_columns[i + 1]}_2")

        else:
            # 如果选择的是均值列，显示直方图
            remaining_columns = [col for col in mean_columns if col != aggregation_field]
            plot_columns = [aggregation_field] + remaining_columns
            
            for i in range(0, len(plot_columns), 2):
                col1, col2 = st.columns(2)  # 每行两个图
                with col1:
                    fig_percentage_histogram = plot_percentage_histogram(filtered_df,
                                                                        "CAMPAIGN ID",
                                                                        top_campaigns,
                                                                        plot_columns[i]
                    )
                    # 为每个图表设置唯一的 key
                    st.plotly_chart(fig_percentage_histogram, key=f"histogram_{plot_columns[i]}_1")
                
                if i + 1 < len(plot_columns):
                    with col2:
                        fig_percentage_histogram = plot_percentage_histogram(filtered_df,
                                                                            "CAMPAIGN ID",
                                                                            top_campaigns,
                                                                            plot_columns[i + 1]
                        )
                        # 为每个图表设置唯一的 key
                        st.plotly_chart(fig_percentage_histogram, key=f"histogram_{plot_columns[i + 1]}_2")
            
            # 接着绘制求和列的饼图
            for i in range(0, len(sum_columns), 2):
                col1, col2 = st.columns(2)  # 创建两列布局
                with col1:
                    fig_aggregation_field = plot_piechart(filtered_df,
                                                        "CAMPAIGN ID",
                                                        top_campaigns,
                                                        other_campaigns,
                                                        sum_columns[i],
                                                        combine_others
                    )
                    # 为每个图表设置唯一的 key
                    st.plotly_chart(fig_aggregation_field, key=f"piechart_{i}")
                
                if i + 1 < len(sum_columns):
                    with col2:
                        fig_aggregation_field = plot_piechart(filtered_df,
                                                            "CAMPAIGN ID",
                                                            top_campaigns,
                                                            other_campaigns,
                                                            sum_columns[i + 1],
                                                            combine_others
                        )
                        # 为每个图表设置唯一的 key
                        st.plotly_chart(fig_aggregation_field, key=f"piechart_{i + 1}")
    else:
        st.warning("请检查上传文件格式是否正确")

if selected_page == "关联销售额分析":
    try:
        if not st.session_state["uploaded_campaign_summary"].empty:
            campaign = st.session_state["uploaded_campaign_summary"]
            if not st.session_state["uploaded_sales_report"].empty:
                sales = st.session_state["uploaded_sales_report"]
            # 用户选择日期范围

            date_range = st.date_input(
                "选择需要分析的时间段:",
                value=[sales['DATE'].min(), sales['DATE'].max()],
                min_value=sales['DATE'].min(),
                max_value=sales['DATE'].max(),
            )

            sales = sales[(sales['DATE'] >= date_range[0]) & (sales['DATE'] <= date_range[1])]

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
        else:
            st.warning("请检查是否同时上传了campaign summary和sales report两份Excel文件")
    except Exception:
        st.warning("请检查是否同时上传了 Campaign Performance 和 sales Report 两份文件")


