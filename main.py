import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import warnings

# code to suppress most of warnings and keep the console output clean
warnings.simplefilter("ignore")

# CONFIGURATION + FILE PATH TO EU STATISTA HOUSE PRICING DATASET 
DATA_FILE = "house_pricing.xlsx"

# 2. COLOR PALETTE
COLORS = {
    'primary': '#2c3e50',    # Dark Blue (EU Avg)
    'highlight': '#e67e22',  # Orange (Selected Country)
    'success': '#2ecc71',    # Green (New Dwellings/Positive)
    'danger': '#e74c3c',     # Red (Existing Dwellings/Negative)
    'accent': '#f1c40f',     # Gold (Diamond Marker)
    'fill_green': '#7df5af', # Light Green Fill
    'fill_red': '#e3948c',   # Light Red Fill
    'range_fill': 'rgba(52, 152, 219, 0.3)', # Light Blue Transparent
    'grey': '#bdc3c7'        # Neutral lines
}

# MAP COLOR SCALE
MAP_COLOR_SCALE = [
    [0.0, '#313695'],   # Deep Blue (Crash)
    [0.25, "#b3f0f4"],  # Light Blue (Drop)
    [0.3, '#ffffff'],   # White (Stable)
    [0.35, "#f4f4a7"],  # Light Yellow (Growth)
    [1.0, '#a50026']    # Deep Red (Overheating)
]
MAP_RANGE = [-10, 25]

# REUSABLE STYLES
CARD_STYLE = {"boxShadow": "0 2px 4px 0 rgba(0,0,0,0.1)", "borderRadius": "5px", "marginBottom": "0px"}

HIGHLIGHT_MARKER_STYLE = dict(
    color=COLORS['accent'], 
    size=20, 
    symbol='diamond', 
    line=dict(width=2, color='black')
)

# DATA ETL PROCESSING

def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        if 'Data' in xls.sheet_names:
            df_raw = pd.read_excel(xls, sheet_name='Data', header=None)
        else:
            raise ValueError("Sheet 'Data' not found in Excel file.")
    except Exception as e:
        raise ValueError(f"Could not open Excel file: {e}")

    #Helper: Find start rows
    def find_start_row(df, search_term):
        matches = df[df.apply(lambda row: row.astype(str).str.contains(search_term).any(), axis=1)]
        if not matches.empty:
            idx = matches.index[0]
            for i in range(idx, idx + 15):
                if "TIME" in str(df.iloc[i].values):
                    return i
        return -1

    row_new = find_start_row(df_raw, "Purchases of newly built dwellings")
    row_exist = find_start_row(df_raw, "Purchases of existing dwellings")
    row_total = find_start_row(df_raw, "Total")
    
    if row_total == -1: 
        time_rows = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains('TIME').any(), axis=1)].index
        if len(time_rows) > 0: row_total = time_rows[0]

    # Helper: Extract Block
    def extract_block(start_row, label_type):
        if start_row == -1: return pd.DataFrame()
        chunk = df_raw.iloc[start_row:]
        chunk.columns = chunk.iloc[0] 
        chunk = chunk[1:] 
        chunk.rename(columns={chunk.columns[0]: 'Country'}, inplace=True)
        
        year_cols = [c for c in chunk.columns if str(c).strip().replace('.0','').isdigit() and len(str(c).strip()) == 4]
        if not year_cols: return pd.DataFrame()

        df_long = pd.melt(chunk, id_vars=['Country'], value_vars=year_cols, var_name='Year', value_name='Value')
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
        df_long['Year'] = pd.to_numeric(df_long['Year'].astype(str).str.replace('.0', '', regex=False))
        df_long.dropna(subset=['Country'], inplace=True)
        df_long['Country'] = df_long['Country'].astype(str).str.strip()
        df_long['Type'] = label_type
        return df_long

    df_total = extract_block(row_total, 'Total')
    df_new = extract_block(row_new, 'New')
    df_exist = extract_block(row_exist, 'Existing')
# REMOVED: TK, CH, UK AS NOT ENOUGH DATA AVAILABLE OR NOT EU ANYMORE
    countries_to_keep = [
        'Belgium', 'Bulgaria', 'Czechia', 'Denmark', 'Germany', 'Estonia', 'Ireland', 
        'Greece', 'Spain', 'France', 'Croatia', 'Italy', 'Cyprus', 'Latvia', 'Lithuania', 
        'Luxembourg', 'Hungary', 'Malta', 'Netherlands', 'Austria', 'Poland', 'Portugal', 
        'Romania', 'Slovenia', 'Slovakia', 'Finland', 'Sweden', 'Iceland', 'Norway'
    ] 
    
    def process_growth_data(df):
        if df.empty: return df
        mask = df['Country'].isin(countries_to_keep)
        aggregates_search = ['European Union', 'EU27', 'Euro area']
        mask_agg = df['Country'].apply(lambda x: any(agg in str(x) for agg in aggregates_search))
        df_filtered = df[mask | mask_agg].copy()
        
        iso_map = {
            'Greece': 'GRC', 'EL': 'GRC', 'United Kingdom': 'GBR', 'UK': 'GBR',
            'Belgium': 'BEL', 'Bulgaria': 'BGR', 'Czechia': 'CZE', 'Denmark': 'DNK',
            'Germany': 'DEU', 'Estonia': 'EST', 'Ireland': 'IRL', 'Spain': 'ESP',
            'France': 'FRA', 'Croatia': 'HRV', 'Italy': 'ITA', 'Cyprus': 'CYP',
            'Latvia': 'LVA', 'Lithuania': 'LTU', 'Luxembourg': 'LUX', 'Hungary': 'HUN',
            'Malta': 'MLT', 'Netherlands': 'NLD', 'Austria': 'AUT', 'Poland': 'POL',
            'Portugal': 'PRT', 'Romania': 'ROU', 'Slovenia': 'SVN', 'Slovakia': 'SVK',
            'Finland': 'FIN', 'Sweden': 'SWE', 'Iceland': 'ISL', 'Norway': 'NOR',
            'Switzerland': 'CHE'
        }
        df_filtered['iso_alpha'] = df_filtered['Country'].map(iso_map)
        df_filtered.loc[df_filtered['iso_alpha'].isna() & df_filtered['Country'].str.contains('Greece'), 'iso_alpha'] = 'GRC'
        
        df_annual = df_filtered.groupby(['Country', 'Year', 'iso_alpha', 'Type'], as_index=False)['Value'].mean()
        df_annual['Index'] = df_annual['Value']

        df_annual.sort_values(['Country', 'Year'], inplace=True)
        df_annual['Value'] = df_annual.groupby('Country')['Value'].pct_change() * 100
        df_annual['Value'] = df_annual['Value'].fillna(0)
        
        return df_annual

    df_total = process_growth_data(df_total)
    df_new = process_growth_data(df_new)
    df_exist = process_growth_data(df_exist)

    return df_total, df_new, df_exist

# INIT.
try:
    df_total, df_new, df_exist = load_and_clean_data(DATA_FILE)
    AVAILABLE_YEARS = sorted(df_total['Year'].unique())
    if AVAILABLE_YEARS:
        MIN_YEAR, MAX_YEAR = min(AVAILABLE_YEARS), max(AVAILABLE_YEARS)
    else:
        MIN_YEAR, MAX_YEAR = 2015, 2024
    
    aggregates_search = ['European Union', 'EU27', 'Euro area']
    mask_eu = df_total['Country'].apply(lambda x: any(agg in str(x) for agg in aggregates_search))
    df_eu_total = df_total[mask_eu].copy()
    df_countries_total = df_total[~mask_eu & df_total['iso_alpha'].notna()].copy()
    
except Exception as e:
    print(f"Data Load Error: {e}")
    df_countries_total, df_eu_total = pd.DataFrame(), pd.DataFrame()
    df_new, df_exist = pd.DataFrame(), pd.DataFrame()
    MIN_YEAR, MAX_YEAR = 2015, 2024
    AVAILABLE_YEARS = []
# APP LAYOUT:
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA], suppress_callback_exceptions=True)

app.layout = dbc.Container([
    dcc.Store(id='store-country', data='Germany'),

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("EU Real Estate", className="fw-bold mb-0"),
            html.Small("Annual House Price Growth (%)", className="text-muted")
        ], width=4, className="d-flex flex-column justify-content-center"),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Select Year:", className="fw-bold mb-0"),
                    dcc.Slider(
                        id='year-slider', min=MIN_YEAR, max=MAX_YEAR, value=MAX_YEAR,
                        marks={str(y): str(y) for y in AVAILABLE_YEARS}, step=None,
                        className="p-0"
                    )
                ], className="p-2") 
            ], style=CARD_STYLE)
        ], width=8)
    ], className="my-2 align-items-center"),

    # KPI Row
    dbc.Row(id='kpi-row', className="mb-3"),

    # Content MAP
    dbc.Row([
        # LEFT: MAP
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Annual Growth Map", className="py-2 fw-bold"),
                dbc.CardBody(
                    dcc.Graph(id='map-graph', style={'height': '700px'}), 
                    className="p-0"
                )
            ], style=CARD_STYLE, className="h-100")
        ], lg=8, className="pe-1"),

        # RIGHT: TABS
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    dcc.Tabs(id='analysis-tabs', value='tab-ranking', children=[
                        dcc.Tab(label='Rankings', value='tab-ranking', className="fw-bold"),
                        dcc.Tab(label='Market Overview', value='tab-overview', className="fw-bold"),
                        dcc.Tab(label='Sector Analysis', value='tab-sectors', className="fw-bold"),
                    ], colors={"border": "#d6d6d6", "primary": COLORS['highlight'], "background": "#f8f9fa"}),
                    className="p-0 border-bottom-0"
                ),
                dbc.CardBody(
                    html.Div(id='tabs-content'), 
                    className="p-1",
                    style={'height': '700px'} 
                )
            ], style=CARD_STYLE, className="h-100")
        ], lg=4, className="ps-1")

    ], className="g-0"),

], fluid=True, style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'padding': '10px'})

# CALLBACKS:
# KPI UPDATE
@app.callback(
    Output('kpi-row', 'children'),
    Input('store-country', 'data')
)
def update_kpi_row(country):
    if df_countries_total.empty or country is None:
        return []

    dff = (
        df_countries_total[df_countries_total['Country'] == country]
        .sort_values('Year')
    )
    if dff.empty:
        return []

    # Base year
    start_row = dff[dff['Year'] == 2015]
    start_row = start_row.iloc[0] if not start_row.empty else dff.iloc[0]

    # Latest year
    end_row = dff.iloc[-1]

    start_val = start_row['Index']
    end_val = end_row['Index']

    total_change = ((end_val - start_val) / start_val) * 100
    color = "success" if total_change > 0 else "danger"
    arrow = "▲" if total_change > 0 else "▼"

    return [
        # BASE INDEX KPI
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6(
                        f"Base Index – {country} ({start_row['Year']})",
                        className="text-muted mb-1"
                    ),
                    html.H3(
                        f"{start_val:.1f}",
                        className="fw-bold text-secondary"
                    )
                ], className="p-3 text-center")
            ], style=CARD_STYLE)
        ], width=4),

        # TOTAL CHANGE KPI
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6(
                        f"Total Change – {country} ({start_row['Year']}–{end_row['Year']})",
                        className="text-muted mb-1"
                    ),
                    html.H3(
                        f"{arrow} {total_change:.1f}%",
                        className=f"fw-bold text-{color}"
                    )
                ], className="p-3 text-center")
            ], style=CARD_STYLE)
        ], width=4),

        # CURRENT INDEX KPI
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6(
                        f"Current Index – {country} ({end_row['Year']})",
                        className="text-muted mb-1"
                    ),
                    html.H3(
                        f"{end_val:.1f}",
                        className="fw-bold text-primary"
                    )
                ], className="p-3 text-center")
            ], style=CARD_STYLE)
        ], width=4),
    ]

# UPDATE SELECTION from map or lollipops
@app.callback(
    Output('store-country', 'data'),
    [Input('map-graph', 'clickData'), Input('ranking-graph', 'clickData')], 
    State('store-country', 'data')
)
def update_selection(map_clk, rank_clk, current):
    ctx = callback_context
    if not ctx.triggered: return current
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'map-graph' and map_clk:
        iso = map_clk['points'][0]['location']
        found = df_countries_total[df_countries_total['iso_alpha'] == iso]
        if not found.empty: return found['Country'].iloc[0]

    elif trigger_id == 'ranking-graph' and rank_clk:
        return rank_clk['points'][0]['y'] 

    return current

# UPDATE MAP
@app.callback(
    Output('map-graph', 'figure'),
    Input('year-slider', 'value'),
    Input('store-country', 'data')
)
def update_map(year, country):
    if df_countries_total.empty: return go.Figure()

    dff = df_countries_total[df_countries_total['Year'] == year].copy()
    
    fig_map = px.choropleth(
        dff, 
        locations='iso_alpha', 
        color='Value', 
        hover_name='Country',
        color_continuous_scale=MAP_COLOR_SCALE, 
        range_color=MAP_RANGE,
        hover_data={'iso_alpha': False, 'Value': True},
        labels={'Value': 'Annual Growth (%)'} 
    )
    
    fig_map.update_traces(hovertemplate="<b>%{hovertext}</b><br>Annual Growth: %{z:.1f}%<extra></extra>")
    
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}, 
        geo_bgcolor='rgba(0,0,0,0)',
        geo=dict(scope='europe', projection_scale=0.9, center=dict(lat=54, lon=15), resolution=50),
        coloraxis_colorbar=dict(title="Growth %", thickness=15, len=0.7)
    )

    dff_highlight = df_countries_total[df_countries_total['Country'] == country]
    if not dff_highlight.empty:
        iso = dff_highlight['iso_alpha'].iloc[0]
        fig_map.add_trace(go.Choropleth(
            locations=[iso], z=[1], locationmode='ISO-3',
            colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']], showscale=False,
            marker_line_color='black', marker_line_width=3,
            hoverinfo='skip'
        ))
    return fig_map

# RENDER TABS
@app.callback(
    Output('tabs-content', 'children'),
    [Input('analysis-tabs', 'value'), Input('year-slider', 'value'), Input('store-country', 'data')]
)
def render_content(tab, year, country):
    if df_countries_total.empty: return html.Div("No Data")

    # TAB FOR RANKING ---
    if tab == 'tab-ranking':
        dff = df_countries_total[df_countries_total['Year'] == year].sort_values('Value', ascending=True)

        cols_markers = [COLORS['highlight'] if x == country else '#4bef8f' for x in dff['Country']]

        fig_risers = go.Figure()
        fig_risers.add_trace(go.Scatter(
            x=dff['Value'], y=dff['Country'], mode='markers',
            marker=dict(size=10, color=cols_markers), showlegend=False, hoverinfo='x+y'
        ))
        for _, row in dff.iterrows():
            fig_risers.add_trace(go.Scatter(
                x=[0, row['Value']], y=[row['Country'], row['Country']],
                mode='lines', line=dict(color=COLORS['grey'], width=3), showlegend=False, hoverinfo='skip'
            ))

        fig_risers.update_layout(
            title=f"Market Ranking ({year})", template='plotly_white', 
            xaxis=dict(title='Annual Growth (%)'), yaxis=dict(type='category'),
            margin=dict(l=0, r=10, t=40, b=30), height=680 
        )
        return dcc.Graph(id='ranking-graph', figure=fig_risers, style={'height': '100%'})

    # OVERVIEW graph
    elif tab == 'tab-overview':
        avg_series = df_countries_total.groupby('Year')['Value'].mean()
        min_series = df_countries_total.groupby('Year')['Value'].min()
        max_series = df_countries_total.groupby('Year')['Value'].max()
        
        fig_overview = go.Figure()
        fig_overview.add_trace(go.Scatter(x=avg_series.index, y=max_series, mode='lines', line=dict(width=0), showlegend=False))
        fig_overview.add_trace(go.Scatter(
            x=avg_series.index, y=min_series, mode='lines', line=dict(width=0), 
            fill='tonexty', fillcolor=COLORS['range_fill'], name='Market Range'
        ))
        fig_overview.add_trace(go.Scatter(
            x=avg_series.index, y=avg_series, mode='lines', 
            name='EU Average', line=dict(color=COLORS['primary'], width=2, dash='dash')
        ))

        country_data = df_countries_total[df_countries_total['Country'] == country].sort_values('Year')
        if not country_data.empty:
            fig_overview.add_trace(go.Scatter(
                x=country_data['Year'], y=country_data['Value'],
                mode='lines+markers', name=country,
                line=dict(color=COLORS['highlight'], width=4, shape='spline')
            ))

        fig_overview.add_vline(x=year, line_dash="dot", line_color="gray")
        fig_overview.update_layout(
            title=f"{country} vs EU Market Benchmark", template='plotly_white',
            yaxis=dict(title='Annual Growth (%)'), legend=dict(orientation="h", y=1.02),
            margin=dict(l=0, r=10, t=40, b=30), height=680
        )
        return dcc.Graph(figure=fig_overview, style={'height': '100%'})

    # SECTORS new and existing properties representation with Candlestick (Refactored & Hover Simplified)
    elif tab == 'tab-sectors':
        
        box_new = df_new[df_new['Year'] == year]
        box_exist = df_exist[df_exist['Year'] == year]

        fig_box = go.Figure()
        
        # New Dwellings from xls file, which means new Properties (Hover: Max/Mean/Min)
        fig_box.add_trace(go.Box(
            y=box_new['Value'], 
            name='New Properties', 
            marker_color=COLORS['success'], 
            fillcolor=COLORS['fill_green'], 
            boxpoints=False,  
            boxmean=True,
            # HOVER TEMPLATE HERE
            hovertemplate="<b>New Properties (EU)</b><br>Max: %{max:.1f}%<br>Mean: %{mean:.1f}%<br>Min: %{min:.1f}%<extra></extra>"
        ))
        
        # Existing Dwellings from xls file, which are the existing properties (Hover: Max/Mean/Min)
        fig_box.add_trace(go.Box(
            y=box_exist['Value'], 
            name='Existing Properties', 
            marker_color=COLORS['danger'], 
            fillcolor=COLORS['fill_red'], 
            boxpoints=False, 
            boxmean=True,
            # HOVER TEMPLATE
            hovertemplate="<b>Existing Properties (EU)</b><br>Max: %{max:.1f}%<br>Mean: %{mean:.1f}%<br>Min: %{min:.1f}%<extra></extra>"
        ))

        # Helper for markers
        def add_country_marker(df_source, label_name, label_suffix):
            val = df_source[df_source['Country'] == country]['Value']
            if not val.empty:
                fig_box.add_trace(go.Scatter(
                    x=[label_name], y=val.values, 
                    mode='markers', 
                    marker=HIGHLIGHT_MARKER_STYLE, 
                    name=f"{country} ({label_suffix})",
                    hovertemplate=f"<b>{country} ({label_suffix})</b><br>Growth: %{{y:.1f}}%<extra></extra>"
                ))

        add_country_marker(box_new, 'New Properties', 'New')
        add_country_marker(box_exist, 'Existing Properties', 'Ex')

        fig_box.update_layout(
            title=f"Where does {country} sit in the European Market? ({year})", 
            template='plotly_white', 
            showlegend=False, 
            yaxis=dict(title='Year-over-Year Price Change (%)'),
            margin=dict(l=0, r=10, t=40, b=30), 
            height=680 
        )

        return dcc.Graph(figure=fig_box, style={'height': '100%'})

if __name__ == '__main__':
    app.run(debug=True)