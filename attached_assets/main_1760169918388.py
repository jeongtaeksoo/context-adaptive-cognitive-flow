#!/usr/bin/env python3
"""
Main entry point for Context-Adaptive Cognitive Flow System

This research simulation implements a 4-stage context-adaptive cognitive flow
designed to enhance cognitive activation in older adults through multi-agent AI personas.

For academic purposes, this file serves as the explicit entry point.
The actual application logic is in app.py (Streamlit web interface).

Usage:
    python main.py              # Run with default settings
    streamlit run app.py        # Alternative direct Streamlit execution
"""

import subprocess
import sys
import os

def main():
    """
    Launch the Streamlit application
    
    This function starts the Streamlit web server on port 5000 with appropriate
    configuration for deployment and research demonstration.
    """
    # Build streamlit command
    cmd = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        'app.py',
        '--server.port',
        '5000',
        '--server.address',
        '0.0.0.0',
        '--server.headless',
        'true'
    ]
    
    print("=" * 70)
    print("Context-Adaptive Cognitive Flow System")
    print("=" * 70)
    print("\nStarting Streamlit application...")
    print(f"Server will be available at: http://0.0.0.0:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    try:
        # Run streamlit
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n\nError starting server: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n\nError: Streamlit not found. Please install dependencies:")
        print("  pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
