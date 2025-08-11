"""
Agent Evolve Dashboard - Main Page

A Streamlit dashboard for visualizing agent evolution progress and trace data.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from agent_evolve.shared_sidebar import render_sidebar

# Load environment variables from current working directory
load_dotenv()

st.set_page_config(
    page_title="Agent Evolve Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render shared sidebar and get database path
db_path = render_sidebar()

# Main page content
st.title("🧬 Agent Evolve Dashboard")

st.markdown("""
Welcome to the Agent Evolve Dashboard! This dashboard provides insights into your traced application.

## Features

- **📝 Prompts**: View and manage all detected prompts in your codebase
- **💻 Code**: Analyze functions and their usage patterns

## Navigation

Use the sidebar to navigate between different sections of the dashboard.

## Getting Started

1. Make sure your application is running with tracing enabled
2. Navigate to the **Prompts** or **Code** pages using the sidebar
3. Select items for evolution and use the evolution buttons

## Database Status

Your database connection status is shown in the sidebar. If the database is not found, make sure:
- Your traced application has been run at least once
- The database path is correct
- You have proper permissions to access the database file
""")