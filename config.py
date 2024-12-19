# 配置模块：config.py

# 页面选项
PAGES = ["数据导入", "数据预览", "趋势分析", "百分比分布", "关联销售额分析"]

# 数值列配置
NUMERIC_COLUMNS = ['TOTAL SALES', 'IMPRESSIONS', 'CLICKS', 'SPEND', 'CPC', 'CPM', 'CTR', 'ROAS']

# 单位映射配置
UNITS_MAPPING = {
    'TOTAL SALES': 'USD',
    'IMPRESSIONS': 'Count',
    'CLICKS': 'Count',
    'SPEND': 'USD',
    'CPC': 'USD',
    'CPM': 'USD',
    'CTR': 'Percentage (%)',
    'ROAS': 'Percentage (%)'
}