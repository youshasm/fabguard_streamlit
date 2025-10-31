"""
FabGuard Defect Detection System Runner
This script starts the Streamlit app on port 8080
"""

import subprocess
import sys
import os

def run_streamlit_app():
    """Run the Streamlit app on port 8080"""
    try:
        # Change to the directory containing the app
        app_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(app_dir)
        
        # Run streamlit with port 8080
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8080",
            "--server.address", "0.0.0.0"
        ]
        
        print("🏭 Starting FabGuard Defect Detection System...")
        print(f"🚀 Running on: http://localhost:8080")
        print("Press Ctrl+C to stop the server")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    run_streamlit_app()