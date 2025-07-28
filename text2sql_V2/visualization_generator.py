"""
시각화 생성 모듈
차트, 그래프, 표 등 다양한 시각화를 생성하는 모듈
"""

import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import base64
import io
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ChartType(Enum):
    """차트 유형"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    AREA = "area"
    WATERFALL = "waterfall"
    FUNNEL = "funnel"

class OutputFormat(Enum):
    """출력 형식"""
    HTML = "html"
    PNG = "png"
    SVG = "svg"
    JSON = "json"
    BASE64 = "base64"

@dataclass
class VisualizationResult:
    """시각화 결과"""
    success: bool
    chart_type: str
    output_format: str
    content: Union[str, bytes, Dict]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class SmartChartGenerator:
    """지능형 차트 생성기"""
    
    def __init__(self, default_theme: str = "plotly_white"):
        """
        초기화
        
        Args:
            default_theme: 기본 테마
        """
        self.default_theme = default_theme
        self.color_palettes = {
            'default': px.colors.qualitative.Set3,
            'professional': px.colors.qualitative.Set1,
            'modern': px.colors.qualitative.Pastel,
            'dark': px.colors.qualitative.Dark24
        }
        
        # 한글 폰트 설정 (matplotlib)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        logger.info("차트 생성기 초기화 완료")
    
    def generate_chart(self, 
                      data: pd.DataFrame,
                      chart_config: Dict[str, Any]) -> VisualizationResult:
        """
        차트 생성
        
        Args:
            data: 데이터프레임
            chart_config: 차트 설정
            
        Returns:
            VisualizationResult: 시각화 결과
        """
        try:
            if data.empty:
                return VisualizationResult(
                    success=False,
                    chart_type="unknown",
                    output_format="html",
                    content="",
                    metadata={},
                    error_message="데이터가 비어있습니다."
                )
            
            chart_type = chart_config.get('type', 'bar')
            output_format = chart_config.get('output_format', 'html')
            
            # 차트 유형에 따른 생성
            if chart_type == ChartType.LINE.value:
                fig = self._create_line_chart(data, chart_config)
            elif chart_type == ChartType.BAR.value:
                fig = self._create_bar_chart(data, chart_config)
            elif chart_type == ChartType.PIE.value:
                fig = self._create_pie_chart(data, chart_config)
            elif chart_type == ChartType.SCATTER.value:
                fig = self._create_scatter_plot(data, chart_config)
            elif chart_type == ChartType.HISTOGRAM.value:
                fig = self._create_histogram(data, chart_config)
            elif chart_type == ChartType.HEATMAP.value:
                fig = self._create_heatmap(data, chart_config)
            else:
                # 데이터 특성에 따른 자동 차트 선택
                fig = self._auto_generate_chart(data, chart_config)
            
            # 공통 레이아웃 적용
            self._apply_common_layout(fig, chart_config)
            
            # 출력 형식에 따른 변환
            content, metadata = self._convert_output(fig, output_format, chart_config)
            
            return VisualizationResult(
                success=True,
                chart_type=chart_type,
                output_format=output_format,
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"차트 생성 실패: {e}")
            return VisualizationResult(
                success=False,
                chart_type=chart_config.get('type', 'unknown'),
                output_format=chart_config.get('output_format', 'html'),
                content="",
                metadata={},
                error_message=str(e)
            )
    
    def _create_line_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """라인 차트 생성"""
        x_col = config.get('x_axis') or data.columns[0]
        y_col = config.get('y_axis') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        fig = go.Figure()
        
        # 그룹별 처리
        group_by = config.get('group_by')
        if group_by and group_by in data.columns:
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group]
                fig.add_trace(go.Scatter(
                    x=group_data[x_col],
                    y=group_data[y_col],
                    mode='lines+markers' if config.get('show_markers', True) else 'lines',
                    name=str(group),
                    line=dict(width=config.get('line_width', 2))
                ))
        else:
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers' if config.get('show_markers', True) else 'lines',
                name=y_col,
                line=dict(width=config.get('line_width', 2))
            ))
        
        # 영역 채우기
        if config.get('fill_area', False):
            fig.update_traces(fill='tonexty')
        
        return fig
    
    def _create_bar_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """막대 차트 생성"""
        x_col = config.get('x_axis') or data.columns[0]
        y_col = config.get('y_axis') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        # 데이터 정렬
        if config.get('sort_by') == 'value':
            data = data.sort_values(y_col, ascending=config.get('ascending', False))
        elif config.get('sort_by') == 'name':
            data = data.sort_values(x_col)
        
        orientation = config.get('orientation', 'vertical')
        
        if orientation == 'horizontal':
            fig = go.Figure(go.Bar(
                x=data[y_col],
                y=data[x_col],
                orientation='h',
                text=data[y_col] if config.get('show_values', True) else None,
                textposition='auto'
            ))
        else:
            fig = go.Figure(go.Bar(
                x=data[x_col],
                y=data[y_col],
                text=data[y_col] if config.get('show_values', True) else None,
                textposition='auto'
            ))
        
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """파이 차트 생성"""
        labels_col = config.get('labels') or data.columns[0]
        values_col = config.get('values') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        # 상위 N개만 표시 (선택적)
        top_n = config.get('top_n')
        if top_n and len(data) > top_n:
            data_sorted = data.nlargest(top_n, values_col)
            others_sum = data.nsmallest(len(data) - top_n, values_col)[values_col].sum()
            if others_sum > 0:
                others_row = pd.DataFrame({labels_col: ['기타'], values_col: [others_sum]})
                data = pd.concat([data_sorted, others_row], ignore_index=True)
            else:
                data = data_sorted
        
        fig = go.Figure(go.Pie(
            labels=data[labels_col],
            values=data[values_col],
            textinfo='label+percent' if config.get('show_percentages', True) else 'label',
            hole=config.get('hole_size', 0),  # 도넛 차트용
            pull=[0.1 if config.get('explode_max', False) and i == data[values_col].idxmax() else 0 
                  for i in range(len(data))]
        ))
        
        return fig
    
    def _create_scatter_plot(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """산점도 생성"""
        x_col = config.get('x_axis') or data.columns[0]
        y_col = config.get('y_axis') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        # 색상 컬럼 (선택적)
        color_col = config.get('color_by')
        size_col = config.get('size_by')
        
        if color_col and color_col in data.columns:
            fig = px.scatter(
                data, x=x_col, y=y_col, color=color_col,
                size=size_col if size_col and size_col in data.columns else None,
                hover_data=config.get('hover_data', [])
            )
        else:
            fig = go.Figure(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                marker=dict(
                    size=data[size_col] if size_col and size_col in data.columns else 8,
                    opacity=0.7
                )
            ))
        
        # 추세선 추가 (선택적)
        if config.get('show_trendline', False):
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=p(data[x_col]),
                mode='lines',
                name='추세선',
                line=dict(dash='dash')
            ))
        
        return fig
    
    def _create_histogram(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """히스토그램 생성"""
        x_col = config.get('x_axis') or data.columns[0]
        
        fig = go.Figure(go.Histogram(
            x=data[x_col],
            nbinsx=config.get('bins', 20),
            opacity=0.7,
            name=x_col
        ))
        
        # 통계 정보 추가 (선택적)
        if config.get('show_stats', True):
            mean_val = data[x_col].mean()
            fig.add_vline(x=mean_val, line_dash="dash", 
                         annotation_text=f"평균: {mean_val:.2f}")
        
        return fig
    
    def _create_heatmap(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """히트맵 생성"""
        # 숫자형 컬럼만 선택
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("히트맵을 위한 숫자형 데이터가 없습니다.")
        
        # 상관관계 매트릭스 또는 피벗 테이블
        if config.get('correlation', False):
            matrix_data = numeric_data.corr()
        else:
            # 피벗 테이블 생성 시도
            index_col = config.get('index') or data.columns[0]
            columns_col = config.get('columns') or data.columns[1] if len(data.columns) > 1 else None
            values_col = config.get('values') or numeric_data.columns[0]
            
            if columns_col:
                matrix_data = data.pivot_table(
                    index=index_col, 
                    columns=columns_col, 
                    values=values_col, 
                    aggfunc='mean'
                ).fillna(0)
            else:
                matrix_data = numeric_data.corr()
        
        fig = go.Figure(go.Heatmap(
            z=matrix_data.values,
            x=matrix_data.columns,
            y=matrix_data.index,
            colorscale=config.get('colorscale', 'RdYlBu'),
            text=matrix_data.values,
            texttemplate='%{text:.2f}' if config.get('show_values', True) else None
        ))
        
        return fig
    
    def _auto_generate_chart(self, data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """데이터 특성에 따른 자동 차트 생성"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # 데이터 특성 분석
        if len(numeric_cols) >= 2 and len(categorical_cols) == 0:
            # 숫자형 데이터만 있는 경우 - 산점도
            config['type'] = ChartType.SCATTER.value
            return self._create_scatter_plot(data, config)
        
        elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # 범주형 + 숫자형 - 막대 차트
            config['type'] = ChartType.BAR.value
            config['x_axis'] = categorical_cols[0]
            config['y_axis'] = numeric_cols[0]
            return self._create_bar_chart(data, config)
        
        elif len(categorical_cols) >= 1 and len(numeric_cols) == 0:
            # 범주형만 있는 경우 - 카운트 차트
            value_counts = data[categorical_cols[0]].value_counts()
            count_data = pd.DataFrame({
                'category': value_counts.index,
                'count': value_counts.values
            })
            config['x_axis'] = 'category'
            config['y_axis'] = 'count'
            return self._create_bar_chart(count_data, config)
        
        else:
            # 기본값 - 첫 번째 컬럼의 히스토그램
            config['type'] = ChartType.HISTOGRAM.value
            return self._create_histogram(data, config)
    
    def _apply_common_layout(self, fig: go.Figure, config: Dict[str, Any]):
        """공통 레이아웃 적용"""
        title = config.get('title', '')
        x_title = config.get('x_title', config.get('x_axis', ''))
        y_title = config.get('y_title', config.get('y_axis', ''))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=config.get('title_size', 16))
            ),
            xaxis_title=x_title,
            yaxis_title=y_title,
            template=config.get('theme', self.default_theme),
            showlegend=config.get('show_legend', True),
            height=config.get('height', 500),
            width=config.get('width', 800),
            font=dict(
                family=config.get('font_family', 'Arial'),
                size=config.get('font_size', 12)
            )
        )
        
        # 색상 팔레트 적용
        color_palette = config.get('color_palette', 'default')
        if color_palette in self.color_palettes:
            fig.update_traces(
                marker=dict(
                    colorscale=self.color_palettes[color_palette]
                )
            )
    
    def _convert_output(self, fig: go.Figure, output_format: str, config: Dict[str, Any]) -> Tuple[Union[str, bytes, Dict], Dict[str, Any]]:
        """출력 형식 변환"""
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'chart_config': config
        }
        
        if output_format == OutputFormat.HTML.value:
            html_content = fig.to_html(
                include_plotlyjs=config.get('include_plotlyjs', 'cdn'),
                div_id=config.get('div_id', 'chart')
            )
            return html_content, metadata
        
        elif output_format == OutputFormat.JSON.value:
            json_content = fig.to_json()
            return json_content, metadata
        
        elif output_format == OutputFormat.PNG.value:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            return img_bytes, metadata
        
        elif output_format == OutputFormat.SVG.value:
            svg_content = fig.to_image(format="svg", engine="kaleido")
            return svg_content, metadata
        
        elif output_format == OutputFormat.BASE64.value:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            base64_content = base64.b64encode(img_bytes).decode('utf-8')
            return base64_content, metadata
        
        else:
            # 기본값: HTML
            html_content = fig.to_html(include_plotlyjs='cdn')
            return html_content, metadata

class TableGenerator:
    """표 생성기"""
    
    def __init__(self):
        self.default_style = {
            'table': 'border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;',
            'header': 'background-color: #f2f2f2; padding: 12px; text-align: left; border: 1px solid #ddd; font-weight: bold;',
            'cell': 'padding: 12px; text-align: left; border: 1px solid #ddd;',
            'row_even': 'background-color: #f9f9f9;',
            'row_odd': 'background-color: #ffffff;'
        }
    
    def generate_table(self, 
                      data: pd.DataFrame, 
                      config: Dict[str, Any]) -> VisualizationResult:
        """표 생성"""
        try:
            if data.empty:
                return VisualizationResult(
                    success=False,
                    chart_type="table",
                    output_format="html",
                    content="",
                    metadata={},
                    error_message="데이터가 비어있습니다."
                )
            
            output_format = config.get('output_format', 'html')
            
            if output_format == 'html':
                content = self._generate_html_table(data, config)
            elif output_format == 'json':
                content = data.to_json(orient='records', ensure_ascii=False)
            elif output_format == 'csv':
                content = data.to_csv(index=False, encoding='utf-8')
            else:
                content = self._generate_html_table(data, config)
            
            metadata = {
                'row_count': len(data),
                'column_count': len(data.columns),
                'generated_at': datetime.now().isoformat()
            }
            
            return VisualizationResult(
                success=True,
                chart_type="table",
                output_format=output_format,
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"표 생성 실패: {e}")
            return VisualizationResult(
                success=False,
                chart_type="table",
                output_format=config.get('output_format', 'html'),
                content="",
                metadata={},
                error_message=str(e)
            )
    
    def _generate_html_table(self, data: pd.DataFrame, config: Dict[str, Any]) -> str:
        """HTML 표 생성"""
        # 페이징 처리
        max_rows = config.get('max_rows', 100)
        if len(data) > max_rows:
            data = data.head(max_rows)
            show_pagination = True
        else:
            show_pagination = False
        
        # 컬럼 선택
        columns = config.get('columns', list(data.columns))
        if columns:
            data = data[columns]
        
        # 컬럼명 한글화
        column_mapping = config.get('column_mapping', {})
        display_columns = [column_mapping.get(col, col) for col in data.columns]
        
        # HTML 생성
        html_parts = []
        
        # 테이블 시작
        table_style = config.get('table_style', self.default_style['table'])
        html_parts.append(f'<table style="{table_style}">')
        
        # 헤더
        html_parts.append('<thead><tr>')
        header_style = config.get('header_style', self.default_style['header'])
        for col in display_columns:
            html_parts.append(f'<th style="{header_style}">{col}</th>')
        html_parts.append('</tr></thead>')
        
        # 데이터 행
        html_parts.append('<tbody>')
        cell_style = config.get('cell_style', self.default_style['cell'])
        
        for idx, row in data.iterrows():
            row_style = self.default_style['row_even'] if idx % 2 == 0 else self.default_style['row_odd']
            html_parts.append(f'<tr style="{row_style}">')
            
            for value in row:
                # 값 포맷팅
                if pd.isna(value):
                    formatted_value = ''
                elif isinstance(value, (int, float)):
                    if config.get('number_format'):
                        formatted_value = f"{value:{config['number_format']}}"
                    else:
                        formatted_value = str(value)
                else:
                    formatted_value = str(value)
                
                html_parts.append(f'<td style="{cell_style}">{formatted_value}</td>')
            
            html_parts.append('</tr>')
        
        html_parts.append('</tbody>')
        html_parts.append('</table>')
        
        # 페이징 정보
        if show_pagination:
            html_parts.append(f'<p style="margin-top: 10px; font-size: 12px; color: #666;">상위 {max_rows}개 행 표시 (전체 {len(data)}개 행)</p>')
        
        return ''.join(html_parts)

class DashboardGenerator:
    """대시보드 생성기"""
    
    def __init__(self, chart_generator: SmartChartGenerator, table_generator: TableGenerator):
        self.chart_generator = chart_generator
        self.table_generator = table_generator
    
    def generate_dashboard(self, 
                          data_sources: List[Tuple[pd.DataFrame, Dict[str, Any]]], 
                          layout_config: Dict[str, Any]) -> VisualizationResult:
        """대시보드 생성"""
        try:
            dashboard_components = []
            
            for i, (data, component_config) in enumerate(data_sources):
                component_type = component_config.get('type', 'chart')
                
                if component_type == 'chart':
                    result = self.chart_generator.generate_chart(data, component_config)
                elif component_type == 'table':
                    result = self.table_generator.generate_table(data, component_config)
                else:
                    continue
                
                if result.success:
                    dashboard_components.append({
                        'id': f'component_{i}',
                        'type': component_type,
                        'content': result.content,
                        'config': component_config
                    })
            
            # 대시보드 HTML 생성
            dashboard_html = self._generate_dashboard_html(dashboard_components, layout_config)
            
            return VisualizationResult(
                success=True,
                chart_type="dashboard",
                output_format="html",
                content=dashboard_html,
                metadata={
                    'component_count': len(dashboard_components),
                    'layout': layout_config.get('layout', 'grid'),
                    'generated_at': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"대시보드 생성 실패: {e}")
            return VisualizationResult(
                success=False,
                chart_type="dashboard",
                output_format="html",
                content="",
                metadata={},
                error_message=str(e)
            )
    
    def _generate_dashboard_html(self, components: List[Dict], layout_config: Dict[str, Any]) -> str:
        """대시보드 HTML 생성"""
        layout_type = layout_config.get('layout', 'grid')
        columns = layout_config.get('columns', 2)
        
        html_parts = []
        
        # CSS 스타일
        html_parts.append('''
        <style>
        .dashboard-container {
            padding: 20px;
            font-family: Arial, sans-serif;
        }
        .dashboard-title {
            text-align: center;
            margin-bottom: 30px;
            font-size: 24px;
            font-weight: bold;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .dashboard-component {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .component-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        </style>
        ''')
        
        # 대시보드 컨테이너 시작
        html_parts.append('<div class="dashboard-container">')
        
        # 제목
        title = layout_config.get('title', '데이터 대시보드')
        html_parts.append(f'<div class="dashboard-title">{title}</div>')
        
        # 그리드 컨테이너
        if layout_type == 'grid':
            html_parts.append(f'<div class="dashboard-grid" style="grid-template-columns: repeat({columns}, 1fr);">')
        else:
            html_parts.append('<div>')
        
        # 각 컴포넌트 추가
        for component in components:
            html_parts.append('<div class="dashboard-component">')
            
            # 컴포넌트 제목
            comp_title = component['config'].get('title', f"{component['type'].title()} {component['id']}")
            html_parts.append(f'<div class="component-title">{comp_title}</div>')
            
            # 컴포넌트 내용
            html_parts.append(str(component['content']))
            
            html_parts.append('</div>')
        
        html_parts.append('</div>')  # 그리드 종료
        html_parts.append('</div>')  # 컨테이너 종료
        
        return ''.join(html_parts)