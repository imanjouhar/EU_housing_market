Europe's Shifting Real Estate Market: A Data Story with Geo-Spatial Visualization

This is an interactive dashboard built with Dash and Plotly that visualizes the annual rate of change in the House Price Index (HPI) in percentage across Europe. Each EU country can be selected and compared with the rest of Europe. The years can be selected from 2015 to 2024.

The application is designed to tell a data story about the volatility and divergence in European real estate, allowing users to spot regional trends, identify “hot” (positive growth) and “cold” (negative growth) markets, and compare a country’s performance against the EU average.

Features
Interactive Choropleth Map

A map of Europe color-coded with a diverging Red–Blue scale to show positive and negative HPI growth for a selected year. The growth is calculated in percentage per year, compared to previous year. This appears when selecting each EU country.

Historical Trend Line

A line chart showing a selected country’s HPI change over time, plotted against the EU-27 average for context.

Lollipop Chart

A horizontal lollipop chart ranking EU countries by annual HPI growth compared with the previous year, starting from the selected year.

Country vs. EU Comparison

A bar chart directly comparing a selected country’s performance to the EU average for the same year.

Full Interactivity (Cross-Filtering)

Clicking a country on the map or on the “Top 10” chart instantly updates all other graphs to reflect that selection.

Data Source

Provider: Eurostat

Dataset: House Price Index (HPI) – Annual Data (prc_hpi_a)

Local File: The app reads the file
h_pricing_peryear_prc_hpi_a$defaultview_spreadsheet (1).xlsx
(recommended to rename to house_pricing.xlsx)

Important: This file must be in the same directory as app.py for the application to run.

Setup and Installation

These instructions assume you have Python 3.9+ installed.

1. Clone or Download

Download app.py and the data file
h_pricing_peryear_prc_hpi_a$defaultview_spreadsheet (1).xlsx,
and save them into the same project directory.
For simplicity, rename the data file to:

house_pricing.xlsx

2. Create and Activate a Virtual Environment

It is recommended to use a virtual environment.

# Navigate to your project directory
cd path/to/your/project

# Create the environment
python3 -m venv eda_project_workspace

# Activate the environment (macOS/Linux)
source eda_project_workspace/bin/activate

3. Install Required Packages
# Upgrade pip
python -m pip install --upgrade pip

# Install Dash, Pandas, Plotly
python -m pip install dash pandas plotly

How to Run the Application

Make sure your environment is active:

source eda_project_workspace/bin/activate


Then run the Dash app:

python main.py


Your terminal will display a message similar to:

Running on http://127.0.0.1:8050/


Open this URL in your browser to view and interact with the dashboard.
