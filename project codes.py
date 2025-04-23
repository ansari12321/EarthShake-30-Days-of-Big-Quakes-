import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap

# Load and preprocess data
try:
a    df = pd.read_csv(r"C:\Users\adars\OneDrive\Desktop\jupyter projects\earthquake_data.csv")
    print("Dataset loaded successfully. Shape:", df.shape)
    print("Sample of latitude column:", df['latitude'].head().tolist())
    if df.empty or df['latitude'].isna().all():
        print("Error: DataFrame is empty or 'latitude' column contains only NaN values.")
        exit(1)
except FileNotFoundError:
    print("Error: Dataset not found at specified path. Please check the file path.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Preprocessing
df['Station Count'] = df['Station Count'].fillna(0)
df['Azimuth Gap'] = df['Azimuth Gap'].fillna(0)
df['Distance'] = df['Distance'].fillna(0)
df['RMS'] = df['RMS'].fillna(0)
df['horizontalError'] = df['horizontalError'].fillna(0)
df['magError'] = df['magError'].fillna(0)
df['magNst'] = df['magNst'].fillna(0)
df['time'] = pd.to_datetime(df['time'], format='ISO8601', utc=True, errors='coerce')
if df['time'].isna().any():
    print("Warning: Some 'time' values could not be parsed. Rows with invalid 'time' will be dropped.")
    df = df.dropna(subset=['time'])
print("Time column parsed. Shape after time drop:", df.shape)
df['region'] = df['place'].str.extract(r',\s*(.*)').fillna('Unknown')
df['Magnitude_Category'] = df['mag'].apply(
    lambda x: 'Below 2.5' if x < 2.5 else ('2.5 - 4.5' if x <= 4.5 else 'Above 4.5')
)

# Check and create bins only if latitude has valid data
if not df['latitude'].empty and not df['latitude'].isna().all():
    df['lat_bin'] = pd.cut(df['latitude'], bins=20, labels=False)
    df['lon_bin'] = pd.cut(df['longitude'], bins=20, labels=False)
else:
    print("Error: 'latitude' or 'longitude' column is empty or contains only NaN values. Binning skipped.")
    exit(1)

# Initialize Dash app
app = Dash(__name__)

# Gradient background style
app.layout = html.Div([
    # Header with logo and title moved to left
    html.Div([
        html.Img(src="/assets/planet-earth.png", 
                 style={'verticalAlign': 'middle', 'marginRight': '15px', 'width': '60px', 'height': '60px'}),
        html.H1("Earthquake Analytics Dashboard", style={
            'color': '#FF6F61', 'fontSize': '48px', 'padding': '15px 0', 'textShadow': '3px 3px 6px #000',
            'fontWeight': 'bold', 'display': 'inline-block', 'letterSpacing': '1px'
        })
    ], style={
        'background': 'linear-gradient(135deg, #1a1a1a, #2a2a2a)', 'borderBottom': '4px solid #FF6F61',
        'display': 'flex', 'alignItems': 'center', 'padding': '10px 20px', 'justifyContent': 'flex-start'
    }),

    # Main layout with sidebar and content
    html.Div([
        # Sidebar
        html.Div([
            html.H3("Controls", style={'color': '#FF8E53', 'fontSize': '26px', 'marginBottom': '25px', 'textShadow': '1px 1px 3px #000'}),
            html.Label("Select Tab:", style={'color': '#FFCC70', 'fontSize': '16px'}),
            dcc.RadioItems(
                id='tab-selector',
                options=[
                    {'label': 'Overview', 'value': 'overview'},
                    {'label': 'Depth & Magnitude', 'value': 'depth-mag'},
                    {'label': 'Regional Analysis', 'value': 'regional'},
                    {'label': 'Magnitude Breakdown', 'value': 'mag-breakdown'}
                ],
                value='overview',
                style={'color': '#FFCC70', 'marginBottom': '30px', 'fontSize': '14px'}
            ),
            html.Label("Time Range:", style={'color': '#FFCC70', 'fontSize': '16px'}),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=df['time'].min().date(),
                max_date_allowed=df['time'].max().date(),
                start_date=df['time'].min().date(),
                end_date=df['time'].max().date(),
                style={'marginBottom': '30px', 'width': '100%', 'fontSize': '14px'}
            ),
            html.Label("Magnitude Range:", style={'color': '#FFCC70', 'fontSize': '16px'}),
            dcc.RangeSlider(
                id='mag-slider',
                min=df['mag'].min(),
                max=df['mag'].max(),
                step=0.1,
                value=[df['mag'].min(), df['mag'].max()],
                marks={i: str(i) for i in range(int(df['mag'].min()), int(df['mag'].max()) + 1)},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            ),
            html.Label("Region:", style={'color': '#FFCC70', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': 'All', 'value': 'All'}] + [
                    {'label': region, 'value': region} for region in sorted(df['region'].unique())
                ],
                value='All',
                multi=True,
                style={'marginBottom': '30px', 'fontSize': '14px', 'color': '#000000', 'backgroundColor': '#2a2a2a', 'border': '1px solid #FF6F61', 'borderRadius': '5px'}
            ),
            html.Label("Depth Range:", style={'color': '#FFCC70', 'fontSize': '16px'}),
            dcc.RangeSlider(
                id='depth-slider',
                min=df['depth'].min(),
                max=df['depth'].max(),
                step=1,
                value=[df['depth'].min(), df['depth'].max()],
                marks={i: str(i) for i in range(int(df['depth'].min()), int(df['depth'].max()) + 1, 100)},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            ),
            html.Button("Reset Filters", id="btn-reset", style={
                'background': 'linear-gradient(45deg, #FF6F61, #FF8E53)', 'color': '#000000', 'border': 'none', 'padding': '15px 30px',
                'cursor': 'pointer', 'marginTop': '30px', 'width': '100%', 'fontSize': '18px', 'borderRadius': '10px', 'boxShadow': '2px 2px 5px #000'
            }),
            html.Button("Download Data", id="btn-download", style={
                'background': 'linear-gradient(45deg, #FF6F61, #FF8E53)', 'color': '#000000', 'border': 'none', 'padding': '15px 30px',
                'cursor': 'pointer', 'marginTop': '20px', 'width': '100%', 'fontSize': '18px', 'borderRadius': '10px', 'boxShadow': '2px 2px 5px #000'
            }),
            dcc.Download(id="download-data"),
            html.Div([
                html.P("Developed by: Adarsh Kumar", style={'color': '#FFCC70', 'fontSize': '16px', 'marginTop': '30px', 'textAlign': 'center', 'fontWeight': 'bold'}),
                html.P("Data Science Student | Aspiring Data Analyst", style={'color': '#FFCC70', 'fontSize': '14px', 'textAlign': 'center'}),
                html.P("Email: adarshsingh6534@gmail.com", style={'color': '#FFCC70', 'fontSize': '14px', 'textAlign': 'center'})
            ], style={'padding': '20px', 'borderTop': '1px solid #FF6F61', 'marginTop': '40px', 'textAlign': 'center'})
        ], style={
            'width': '25%', 'background': 'linear-gradient(135deg, #2a2a2a, #3a3a3a)', 'padding': '30px', 'height': '100vh',
            'color': 'white', 'overflowY': 'auto', 'borderRight': '3px solid #FF6F61', 'boxShadow': '5px 0 10px rgba(0,0,0,0.5)'
        }),

        # Main content with dynamic layout and graph selector
        html.Div([
            dcc.Dropdown(
                id='graph-selector-dropdown',
                options=[
                    {'label': 'None', 'value': 'none'},
                    {'label': '3D Depth Visualization', 'value': '3d', 'disabled': True},
                    {'label': 'High-Risk Zones', 'value': 'risk', 'disabled': True}
                ],
                value='none',
                style={'width': '200px', 'marginBottom': '20px', 'fontSize': '16px', 'color': '#000000', 'background': 'linear-gradient(45deg, #FF6F61, #FF8E53)', 'border': '1px solid #FF6F61', 'borderRadius': '5px', 'display': 'none'}
            ),
            html.Div(id='tab-content', style={'padding': '30px'})
        ], style={'width': '75%', 'background': 'linear-gradient(135deg, #121212, #1e1e1e)', 'minHeight': '100vh'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'animation': 'fadeIn 1s'})
], style={'background': 'linear-gradient(135deg, #1a1a1a, #2a2a2a, #3a3a3a)', 'minHeight': '100vh'})

# Animation CSS
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css'
})

# Custom colormap from EDA
custom_colors = ['#f2f2f2', '#b87333', '#d2691e', '#ff7f50', '#dda0dd', '#8b0000', '#000000']
custom_cmap = LinearSegmentedColormap.from_list("custom_palette", custom_colors, N=256)

# Helper functions for plots
def create_hotspot_map(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    sampled_df = filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42) if len(filtered_df) > 1000 else filtered_df
    fig = px.scatter_geo(
        sampled_df,
        lat='latitude',
        lon='longitude',
        color='mag',
        size='mag',
        hover_name='place',
        hover_data={'mag': True, 'latitude': True, 'longitude': True, 'time': True},
        projection='natural earth',
        title='Global Earthquake Hotspots',
        color_continuous_scale='OrRd',
        template='plotly_dark'
    )
    fig.update_layout(
        title_font=dict(size=28, color='#FF6F61'),
        title_x=0.5,
        margin=dict(l=0, r=0, t=60, b=0),
        height=600,
        showlegend=True
    )
    return fig

def create_magnitude_trends(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    filtered_df = filtered_df.sort_values(by='time')
    sampled_df = filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42) if len(filtered_df) > 1000 else filtered_df
    fig = px.scatter(
        sampled_df,
        x='time',
        y='mag',
        color='mag',
        color_continuous_scale='Turbo',
        hover_data=['place', 'mag', 'latitude', 'longitude', 'time'],
        title='Earthquake Magnitude Trends Over Time'
    )
    fig.update_layout(
        template='plotly_dark',
        title_font=dict(size=24, color='#FF8E53'),
        xaxis_title='Date',
        yaxis_title='Magnitude',
        title_x=0.5,
        margin=dict(l=50, r=50, t=60, b=50),
        height=500
    )
    return fig

def create_depth_histogram(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=filtered_df['depth'], nbinsx=30, marker_color='#00CED1', opacity=0.8))
    fig.update_layout(
        template='plotly_dark',
        title='Distribution of Earthquake Depths',
        title_font=dict(size=22, color='#FF8E53'),
        xaxis_title='Depth (km)',
        yaxis_title='Frequency',
        title_x=0.5,
        margin=dict(l=50, r=50, t=60, b=50),
        height=500
    )
    return fig

def create_mag_depth_scatter(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    sampled_df = filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42) if len(filtered_df) > 1000 else filtered_df
    fig = px.scatter(
        sampled_df,
        x='depth',
        y='mag',
        color='mag',
        color_continuous_scale='plasma',
        hover_data=['place', 'mag', 'depth'],
        title='Magnitude vs Depth of Earthquakes'
    )
    fig.update_layout(
        template='plotly_dark',
        title_font=dict(size=22, color='#FF6F61'),
        xaxis_title='Depth (km)',
        yaxis_title='Magnitude',
        title_x=0.5,
        margin=dict(l=50, r=50, t=60, b=50),
        height=500
    )
    return fig

def create_3d_depth_plot(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    sampled_df = filtered_df.sample(n=min(500, len(filtered_df)), random_state=42) if len(filtered_df) > 500 else filtered_df
    fig = go.Figure(data=[
        go.Scatter3d(
            x=sampled_df['longitude'],
            y=sampled_df['latitude'],
            z=sampled_df['depth'] * -1,
            mode='markers',
            marker=dict(size=5, color=sampled_df['mag'], colorscale='Viridis', showscale=True),
            text=sampled_df['place'],
            hoverinfo='text+x+y+z'
        )
    ])
    fig.update_layout(
        template='plotly_dark',
        title='3D Earthquake Depth Visualization',
        title_font=dict(size=22, color='#FF6F61'),
        title_x=0.5,
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Depth (km)',
            xaxis=dict(color='white'),
            yaxis=dict(color='white'),
            zaxis=dict(color='white')
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=500
    )
    return fig

def create_region_frequency(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    top_regions = filtered_df['region'].value_counts().head(10).reset_index()
    top_regions.columns = ['region', 'count']
    fig = px.bar(
        top_regions,
        x='count',
        y='region',
        orientation='h',
        title='Top 10 Regions by Earthquake Frequency',
        color='count',
        color_continuous_scale='magma'
    )
    fig.update_layout(
        template='plotly_dark',
        title_font=dict(size=22, color='#FF8E53'),
        xaxis_title='Number of Earthquakes',
        yaxis_title='Region',
        title_x=0.5,
        margin=dict(l=200, r=50, t=60, b=50),
        height=500
    )
    return fig

def create_region_avg_magnitude(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    avg_mag_by_region = filtered_df.groupby('region')['mag'].mean().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(
        avg_mag_by_region,
        x='mag',
        y='region',
        orientation='h',
        title='Top 10 Regions with Highest Average Magnitude',
        color='mag',
        color_continuous_scale='plasma'
    )
    fig.update_layout(
        template='plotly_dark',
        title_font=dict(size=22, color='#FF6F61'),
        xaxis_title='Average Magnitude',
        yaxis_title='Region',
        title_x=0.5,
        margin=dict(l=200, r=50, t=60, b=50),
        height=500
    )
    return fig

def create_risk_zones(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    risk_df = filtered_df.groupby('region').agg({
        'id': 'count',
        'mag': 'mean'
    }).rename(columns={'id': 'quake_count', 'mag': 'avg_magnitude'}).reset_index()
    risk_df['risk_score'] = risk_df['quake_count'] * risk_df['avg_magnitude']
    risk_df = risk_df.sort_values(by='risk_score', ascending=False).head(15)
    fig = px.scatter(
        risk_df,
        x='quake_count',
        y='avg_magnitude',
        size='risk_score',
        color='region',
        hover_data=['risk_score'],
        title='High-Risk Earthquake Zones (Top 15)'
    )
    fig.update_layout(
        template='plotly_dark',
        title_font=dict(size=22, color='#FF6F61'),
        xaxis_title='Number of Earthquakes',
        yaxis_title='Average Magnitude',
        title_x=0.5,
        margin=dict(l=50, r=50, t=60, b=50),
        height=500
    )
    return fig

def create_magnitude_category(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    freq_data = filtered_df['Magnitude_Category'].value_counts().sort_index().reset_index()
    freq_data.columns = ['Category', 'Count']
    fig = px.bar(
        freq_data,
        x='Count',
        y='Category',
        orientation='h',
        title='Earthquake Frequency by Magnitude Category',
        color='Category',
        color_discrete_sequence=['#FF6F61', '#FF8E53', '#FFCC70']
    )
    fig.update_layout(
        template='plotly_dark',
        title_font=dict(size=20, color='#FFCC70'),
        xaxis_title='Number of Earthquakes',
        yaxis_title='Magnitude Category',
        title_x=0.5,
        margin=dict(l=200, r=50, t=60, b=50),
        height=500
    )
    return fig

def create_magnitude_depth_time(filtered_df):
    if filtered_df.empty:
        return go.Figure()
    sampled_df = filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42) if len(filtered_df) > 1000 else filtered_df
    fig = px.scatter(
        sampled_df,
        x='depth',
        y='mag',
        animation_frame=sampled_df['time'].dt.strftime('%Y-%m-%d'),
        animation_group='place',
        color='mag',
        size='mag',
        hover_data=['place', 'time'],
        title='Magnitude and Depth Over Time (Animated)',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        template='plotly_dark',
        title_font=dict(size=20, color='#FFCC70'),
        xaxis_title='Depth (km)',
        yaxis_title='Magnitude',
        title_x=0.5,
        margin=dict(l=50, r=50, t=60, b=50),
        height=500
    )
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 300
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 200
    return fig

# Callback for tab content and graph selector visibility
@app.callback(
    [Output('tab-content', 'children'),
     Output('graph-selector-dropdown', 'style'),
     Output('graph-selector-dropdown', 'options'),
     Output('graph-selector-dropdown', 'value')],
    [Input('tab-selector', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('mag-slider', 'value'),
     Input('region-dropdown', 'value'),
     Input('depth-slider', 'value'),
     Input('graph-selector-dropdown', 'value')]
)
def update_tab(tab, start_date, end_date, mag_range, regions, depth_range, graph_select):
    if not all([start_date, end_date, mag_range, depth_range]):
        return (html.Div(["Error: Missing input data"], style={'color': 'red', 'fontSize': '18px', 'textAlign': 'center'}),
                {'display': 'none'}, [{'label': 'None', 'value': 'none'}], 'none')

    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    print(f"Start Date: {start_date}, End Date: {end_date}, Types: {type(start_date)}, {type(end_date)}")

    filtered_df = df.copy()
    try:
        filtered_df = filtered_df[
            (filtered_df['time'] >= start_date) &
            (filtered_df['time'] <= end_date) &
            (filtered_df['mag'] >= mag_range[0]) &
            (filtered_df['mag'] <= mag_range[1]) &
            (filtered_df['depth'] >= depth_range[0]) &
            (filtered_df['depth'] <= depth_range[1])
        ]
        if regions and regions != ['All']:
            if isinstance(regions, str):
                regions = [regions]
            filtered_df = filtered_df[filtered_df['region'].isin(regions)]
        elif regions == ['All']:
            pass
        print(f"Filtered data shape: {filtered_df.shape}, Time dtype: {filtered_df['time'].dtype}")
        if filtered_df.empty:
            return (html.Div(["No data available for selected filters"], style={'color': 'white', 'fontSize': '18px', 'textAlign': 'center'}),
                    {'display': 'none'}, [{'label': 'None', 'value': 'none'}], 'none')
    except Exception as e:
        return (html.Div([f"Error filtering data: {str(e)}"], style={'color': 'red', 'fontSize': '18px', 'textAlign': 'center'}),
                {'display': 'none'}, [{'label': 'None', 'value': 'none'}], 'none')

    selector_style = {'width': '200px', 'marginBottom': '20px', 'fontSize': '16px', 'color': '#000000', 'background': 'linear-gradient(45deg, #FF6F61, #FF8E53)', 'border': '1px solid #FF6F61', 'borderRadius': '5px', 'display': 'none'}
    selector_options = [{'label': 'None', 'value': 'none'}]

    if tab in ['depth-mag', 'regional']:
        selector_style['display'] = 'block'
        if tab == 'depth-mag':
            selector_options.append({'label': '3D Depth Visualization', 'value': '3d'})
        elif tab == 'regional':
            selector_options.append({'label': 'High-Risk Zones', 'value': 'risk'})
        valid_values = [opt['value'] for opt in selector_options]
        if graph_select not in valid_values:
            graph_select = 'none'

    if tab == 'overview':
        return (html.Div([
            html.Div([
                dcc.Graph(figure=create_hotspot_map(filtered_df), style={'width': '100%', 'display': 'block'})
            ], style={'width': '100%', 'marginBottom': '30px'}),
            html.Div([
                dcc.Graph(figure=create_magnitude_trends(filtered_df), style={'width': '100%', 'display': 'block'}),
                html.Div([
                    html.H4("Quick Stats", style={'color': '#FFCC70', 'fontSize': '22px', 'textShadow': '1px 1px 3px #000'}),
                    html.P(f"Total Quakes: {len(filtered_df)}", style={'color': '#FFCC70', 'fontSize': '18px'}),
                    html.P(f"Max Magnitude: {filtered_df['mag'].max():.1f}" if not filtered_df.empty else "N/A", style={'color': '#FFCC70', 'fontSize': '18px'})
                ], style={'background': 'linear-gradient(135deg, #2a2a2a, #3a3a3a)', 'padding': '20px', 'borderRadius': '15px', 'marginTop': '20px', 'boxShadow': '3px 3px 10px #000'})
            ], style={'width': '100%'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '40px'}),
                selector_style, selector_options, graph_select)
    elif tab == 'depth-mag':
        content = [
            html.Div([
                dcc.Graph(figure=create_depth_histogram(filtered_df), style={'width': '100%', 'display': 'block'})
            ], style={'width': '100%', 'marginBottom': '30px'}),
            html.Div([
                dcc.Graph(figure=create_mag_depth_scatter(filtered_df), style={'width': '100%', 'display': 'block'})
            ], style={'width': '100%', 'marginBottom': '30px'})
        ]
        if graph_select == '3d':
            content.append(html.Div([
                dcc.Graph(figure=create_3d_depth_plot(filtered_df), style={'width': '100%', 'display': 'block'})
            ], style={'width': '100%', 'marginBottom': '30px'}))
        return (html.Div(content, style={'display': 'flex', 'flexDirection': 'column', 'gap': '40px'}),
                selector_style, selector_options, graph_select)
    elif tab == 'regional':
        content = [
            html.Div([
                dcc.Graph(figure=create_region_frequency(filtered_df), style={'width': '100%', 'display': 'block'})
            ], style={'width': '100%', 'marginBottom': '30px'}),
            html.Div([
                dcc.Graph(figure=create_region_avg_magnitude(filtered_df), style={'width': '100%', 'display': 'block'})
            ], style={'width': '100%', 'marginBottom': '30px'})
        ]
        if graph_select == 'risk':
            content.append(html.Div([
                dcc.Graph(figure=create_risk_zones(filtered_df), style={'width': '100%', 'display': 'block'})
            ], style={'width': '100%', 'marginBottom': '30px'}))
        return (html.Div(content, style={'display': 'flex', 'flexDirection': 'column', 'gap': '40px'}),
                selector_style, selector_options, graph_select)
    elif tab == 'mag-breakdown':
        return (html.Div([
            html.Div([
                dcc.Graph(figure=create_magnitude_category(filtered_df), style={'width': '100%', 'display': 'block'})
            ], style={'width': '100%', 'marginBottom': '30px'}),
            html.Div([
                dcc.Graph(figure=create_magnitude_depth_time(filtered_df), style={'width': '100%', 'display': 'block'})
            ], style={'width': '100%', 'marginBottom': '30px'})
        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '40px'}),
                selector_style, selector_options, graph_select)

# Callback for reset button
@app.callback(
    [Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date'),
     Output('mag-slider', 'value'),
     Output('region-dropdown', 'value'),
     Output('depth-slider', 'value')],
    Input('btn-reset', 'n_clicks')
)
def reset_filters(n_clicks):
    if n_clicks:
        return (
            df['time'].min().date(),
            df['time'].max().date(),
            [df['mag'].min(), df['mag'].max()],
            ['All'],
            [df['depth'].min(), df['depth'].max()]
        )
    raise PreventUpdate

# Callback for download button
@app.callback(
    Output("download-data", "data"),
    Input("btn-download", 'n_clicks'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('mag-slider', 'value'),
     Input('region-dropdown', 'value'),
     Input('depth-slider', 'value')]
)
def download_data(n_clicks, start_date, end_date, mag_range, regions, depth_range):
    if n_clicks:
        filtered_df = df.copy()
        start_date = pd.to_datetime(start_date).tz_localize('UTC')
        end_date = pd.to_datetime(end_date).tz_localize('UTC')
        filtered_df = filtered_df[
            (filtered_df['time'] >= start_date) &
            (filtered_df['time'] <= end_date) &
            (filtered_df['mag'] >= mag_range[0]) &
            (filtered_df['mag'] <= mag_range[1]) &
            (filtered_df['depth'] >= depth_range[0]) &
            (filtered_df['depth'] <= depth_range[1])
        ]
        if regions and regions != ['All']:
            if isinstance(regions, str):
                regions = [regions]
            filtered_df = filtered_df[filtered_df['region'].isin(regions)]
        return dcc.send_data_frame(filtered_df.to_csv, "filtered_earthquake_data.csv")
    raise PreventUpdate

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)
