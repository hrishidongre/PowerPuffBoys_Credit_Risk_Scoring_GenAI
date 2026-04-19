"""
Primary Streamlit entrypoint for local runs and cloud deployment.

Run:
    streamlit run app.py
"""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("agent_app.py")), run_name="__main__")
