import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def process_data(data: pd.DataFrame, 
                 column_to_aggregate: str, 
                 min_rank: int, 
                 max_rank: int, 
                 date_range=None):
    """
    处理数据：根据条件过滤并聚合数据，返回适合作图的数据。
    
    参数：
    data (DataFrame): 输入的原始数据
    column_to_aggregate (str): 聚合的字段（如 'ROAS' 或 'CTR'）
    min_rank (int): 排名范围的起始位置
    max_rank (int): 排名范围的结束位置
    date_range (list): 日期范围，默认为 None（表示不做日期过滤）
    
    返回：
    aggregated_summary (DataFrame): 处理后的数据，适合绘图
    total_summary (Series): 根据 column_to_aggregate 排序后的总汇总数据
    """
    # 确保 date_range 有默认值
    if not date_range or len(date_range) != 2:
        date_range = [data['DATE'].min(), data['DATE'].max()]

    # 根据日期范围过滤数据
    filtered_data = data[(data['DATE'] >= date_range[0]) & (data['DATE'] <= date_range[1])]

    # 根据 `column_to_aggregate` 对数据排序
    if column_to_aggregate in ['ROAS', 'CTR']:
        total_summary = filtered_data.groupby('CAMPAIGN ID')[column_to_aggregate].mean().sort_values(ascending=False)
    else:
        total_summary = filtered_data.groupby('CAMPAIGN ID')[column_to_aggregate].sum().sort_values(ascending=False)

    # 获取选中的 Campaign ID
    selected_campaigns = total_summary.iloc[min_rank - 1:max_rank].index 

    # 聚合数据，计算每个 'CAMPAIGN ID' 和 'PLACEMENT TYPE' 的值
    aggregated_summary = (
        filtered_data[filtered_data['CAMPAIGN ID'].isin(selected_campaigns)]
        .groupby(['DATE', 'CAMPAIGN ID', 'PLACEMENT TYPE'])[column_to_aggregate]
        .sum()
        .reset_index()
    )

    return aggregated_summary, total_summary

def plot_linechart(aggregated_summary: pd.DataFrame, 
                   column_to_aggregate: str, 
                   total_summary: pd.Series):
    fig = px.line(
        aggregated_summary,
        x='DATE',
        y=column_to_aggregate,
        color='CAMPAIGN ID',
        line_dash='PLACEMENT TYPE',
        markers=True,
        title=f"Interactive Trend: {column_to_aggregate} by Campaign and Placement Type",
        category_orders={'CAMPAIGN ID': list(total_summary.index)}
    )

    line_dash_map = {'PLA': 'solid', 'Banner': 'dash'}
    fig.for_each_trace(lambda trace: trace.update(line=dict(dash=line_dash_map.get(trace.name.split(', ')[1], 'solid'))))

    fig.update_layout(template='plotly_white', hovermode='x unified')

    return fig, fig.data  # 返回图表和图表数据



def create_comparison_chart(df: pd.DataFrame, selected_campaign_id: str, selected_params: list, units_mapping: dict, date_range=None):
    # 数据按日期汇总
    if not date_range or len(date_range) != 2:
        date_range = [df['DATE'].min(), df['DATE'].max()]

    df = df[(df['DATE'] >= date_range[0]) & (df['DATE'] <= date_range[1])]

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

    return fig_comparison, total_values, ranks


def plot_piechart(data: pd.DataFrame,
                  column_to_aggregate: str,
                  top_campaigns: list,
                  other_campaigns: list,
                  aggregation_column: str,
                  combine_others: bool = True
                ):
    # 过滤数据，只选择 top_campaigns 中的行
    top_summary = data[data[column_to_aggregate].isin(top_campaigns)]
    
    # 按 column_to_aggregate 聚合求和
    top_grouped = top_summary.groupby(column_to_aggregate)[aggregation_column].sum()
    
    # 如果选择合并 "others"，则计算其他campaign的总和
    if combine_others:
        others_sum = data[data[column_to_aggregate].isin(other_campaigns)][aggregation_column].sum()
        top_grouped['others'] = others_sum    

    # 计算百分比
    total_sum = top_grouped.sum()
    percentages = (top_grouped / total_sum) * 100
    
    # 绘制饼图
    fig = px.pie(
        names=top_grouped.index,
        values=top_grouped.values,
        title=f"{aggregation_column} 在周期内的百分比分布分析",
        color=top_grouped.index,
        hole=0.3  # 可选，创建一个环形饼图
    )

    # 使用 update_traces 设置文本信息：显示每个 campaign 的总和和百分比
    fig.update_traces(
        textinfo='label+value+percent',  # 显示标签、总和和百分比
        hovertemplate='%{label}: %{value} (%{percent})'  # 悬停时显示实际值和百分比
    )

    return fig
    
def plot_percentage_histogram(data: pd.DataFrame,
                                  column_to_aggregate: str,
                                  top_campaigns: list,
                                  aggregation_column: str):
    top_summary = data[data[column_to_aggregate].isin(top_campaigns)]
    top_grouped = top_summary.groupby(column_to_aggregate)[aggregation_column].mean().reset_index()
    # 使用 Plotly Express 创建直方图
    fig = px.bar(
        top_grouped,
        x = top_grouped[column_to_aggregate].astype(str),
        y = top_grouped[aggregation_column],
        title=f"{aggregation_column} 在周期内的直方图分析",
        color=top_grouped[column_to_aggregate].astype(str)  # 确保颜色映射也基于字符串
    )

    # 强制设置 x 轴为离散类型
    fig.update_layout(
        xaxis=dict(type="category")
    )

    return fig

# def plot_linechart(data: pd.DataFrame,
#                    column_to_aggregate:str,
#                    min_rank: int,
#                    max_rank: int,
#                    date_range = None):
    
#     # 确保 date_range 有默认值
#     if not date_range or len(date_range) != 2:
#         date_range = [data['DATE'].min(), data['DATE'].max()]

#     filtered_data = data[(data['DATE'] >= date_range[0]) & (data['DATE'] <= date_range[1])]

#     # 根据 `column_to_aggregate` 对数据排序
#     if column_to_aggregate in ['ROAS', 'CTR']:
#         total_summary = filtered_data.groupby('CAMPAIGN ID')[column_to_aggregate].mean().sort_values(ascending=False)
#     else:
#         total_summary = filtered_data.groupby('CAMPAIGN ID')[column_to_aggregate].sum().sort_values(ascending=False)


#     selected_campaigns = total_summary.iloc[min_rank - 1:max_rank].index 

#     aggregated_summary = (
#         filtered_data[filtered_data['CAMPAIGN ID'].isin(selected_campaigns)]
#         .groupby(['DATE', 'CAMPAIGN ID', 'PLACEMENT TYPE'])[column_to_aggregate]
#         .sum()
#         .reset_index()
#     )

#     # 创建折线图
#     fig = px.line(
#         aggregated_summary,
#         x='DATE',
#         y=column_to_aggregate,
#         color='CAMPAIGN ID',
#         line_dash='PLACEMENT TYPE',
#         markers=True,
#         title=f"Interactive Trend: {column_to_aggregate} by Campaign and Placement Type",
#         category_orders={'CAMPAIGN ID': list(total_summary.index)}
#     )
#     # 手动定义线型映射：确保 `PLA` 是实线，`Banner` 是虚线
#     line_dash_map = {'PLA': 'solid', 'Banner': 'dash'}
#     fig.for_each_trace(lambda trace: trace.update(line=dict(dash=line_dash_map.get(trace.name.split(', ')[1], 'solid'))))

#     fig.update_layout(template='plotly_white', hovermode='x unified')

#     return fig