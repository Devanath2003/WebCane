"""
WebCane3 - Web UI Launcher
Run this script to start the web interface.
Usage: python run_web.py
"""

import webbrowser
import threading
import time

def open_browser():
    """Open browser after a short delay to let the server start."""
    time.sleep(2)
    webbrowser.open("http://localhost:5000")

if __name__ == '__main__':
    print("=" * 60)
    print("  WebCane3 - Web UI")
    print("  Starting server at http://localhost:5000")
    print("=" * 60)

    # Open browser in background
    threading.Thread(target=open_browser, daemon=True).start()

    # Start the Flask app
    from web_app import app, socketio
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
