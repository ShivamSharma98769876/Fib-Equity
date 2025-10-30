#!/usr/bin/env python3
"""
Main entry point for the Swing Trade Stock Screener application
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Main entry point"""
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("ERROR: Streamlit not found. Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        import streamlit
    
    # Run the Streamlit app
    app_path = current_dir / "src" / "ui" / "main.py"
    
    if not app_path.exists():
        print(f"ERROR: App file not found: {app_path}")
        return 1
    
    print("Starting Swing Trade Stock Screener...")
    print("Dashboard will open in your browser")
    print("Press Ctrl+C to stop the application")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port", "8503",
        "--server.address", "localhost"
    ])
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
