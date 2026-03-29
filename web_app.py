"""
WebCane3 - Web UI Server
Flask + SocketIO backend that wraps the existing WebCane class.
Streams real-time logs to the browser via WebSocket.
"""

import sys
import io
import threading
import time
import traceback
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'webcane3-ui-secret'
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")

# Global state
webcane_instance = None
session_lock = threading.Lock()
is_running = False


class SocketIOWriter:
    """
    Custom stdout writer that intercepts print() calls
    and emits each line as a WebSocket 'log' event.
    Also writes to the real stdout for debugging.
    Rate-limited to avoid overwhelming the WebSocket transport.
    """
    def __init__(self, original_stdout, socketio_instance):
        self.original = original_stdout
        self.sio = socketio_instance
        self.buffer = ""
        self._last_emit = 0
        self._MIN_EMIT_INTERVAL = 0.01  # 10ms min between emits

    def _rate_limited_emit(self, line):
        """Emit a log line with rate limiting."""
        try:
            now = time.time()
            elapsed = now - self._last_emit
            if elapsed < self._MIN_EMIT_INTERVAL:
                time.sleep(self._MIN_EMIT_INTERVAL - elapsed)
            self.sio.emit('log', {'message': line}, namespace='/')
            self._last_emit = time.time()
        except Exception:
            pass

    def write(self, text):
        # Always write to real stdout
        self.original.write(text)

        if not text:
            return

        # Buffer partial writes and emit complete lines
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.rstrip('\r')
            if line:
                self._rate_limited_emit(line)

        # Handle lines that end with \r but no \n (like progress)
        if '\r' in self.buffer:
            line = self.buffer.rstrip('\r')
            self.buffer = ""
            if line:
                self._rate_limited_emit(line)

    def flush(self):
        self.original.flush()
        # Flush any remaining buffer
        if self.buffer.strip():
            try:
                self.sio.emit('log', {'message': self.buffer.strip()}, namespace='/')
            except Exception:
                pass
            self.buffer = ""

    def fileno(self):
        return self.original.fileno()

    def isatty(self):
        return False


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve the main UI page."""
    return render_template('index.html')


# ==================== SOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Client connected."""
    emit('status', {'state': 'connected', 'message': 'Connected to WebCane3 server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected."""
    pass


@socketio.on('start_session')
def handle_start_session(data):
    """Initialize WebCane with selected settings."""
    global webcane_instance, is_running

    if webcane_instance is not None:
        emit('status', {'state': 'error', 'message': 'Session already active. Stop it first.'})
        return

    model = data.get('model', 'deepseek')
    vlm_only = data.get('vlm_only', False)

    emit('status', {'state': 'initializing', 'message': 'Starting WebCane3...'})
    emit('log', {'message': '━━━ Initializing WebCane3 ━━━'})

    def init_webcane():
        global webcane_instance
        # Redirect stdout to capture print output
        original_stdout = sys.stdout
        sys.stdout = SocketIOWriter(original_stdout, socketio)

        try:
            from Webcane3.main import WebCane
            webcane_instance = WebCane(
                supervisor_model=model,
                vlm_only_mode=vlm_only
            )
            socketio.emit('status', {
                'state': 'ready',
                'message': 'WebCane3 ready! Enter a goal to execute.'
            })
            socketio.emit('session_started', {'success': True})
        except Exception as e:
            traceback.print_exc()
            socketio.emit('status', {
                'state': 'error',
                'message': f'Failed to initialize: {str(e)}'
            })
            socketio.emit('session_started', {'success': False, 'error': str(e)})
            webcane_instance = None
        finally:
            sys.stdout = original_stdout

    socketio.start_background_task(init_webcane)


@socketio.on('execute_goal')
def handle_execute_goal(data):
    """Execute a goal using the WebCane instance."""
    global webcane_instance, is_running

    if webcane_instance is None:
        emit('status', {'state': 'error', 'message': 'No active session. Start one first.'})
        return

    if is_running:
        emit('status', {'state': 'error', 'message': 'A goal is already running. Please wait.'})
        return

    goal = data.get('goal', '').strip()
    if not goal:
        emit('status', {'state': 'error', 'message': 'Please enter a goal.'})
        return

    is_running = True
    emit('status', {'state': 'running', 'message': f'Executing: {goal}'})
    emit('log', {'message': f'━━━ Goal: {goal} ━━━'})

    def run_goal():
        global is_running
        original_stdout = sys.stdout
        sys.stdout = SocketIOWriter(original_stdout, socketio)

        try:
            result = webcane_instance.execute_goal(goal)

            socketio.emit('goal_result', {
                'success': result.get('success', False),
                'actions_taken': result.get('actions_taken', 0),
                'successful_actions': result.get('successful_actions', 0),
                'elapsed_time': round(result.get('elapsed_time', 0), 2),
                'final_url': result.get('final_url', 'N/A'),
                'error': result.get('error')
            })

            if result.get('success'):
                socketio.emit('status', {
                    'state': 'ready',
                    'message': 'Goal completed successfully!'
                })
            else:
                error_msg = result.get('error', 'Unknown error')
                socketio.emit('status', {
                    'state': 'ready',
                    'message': f'Goal failed: {error_msg}'
                })

        except Exception as e:
            traceback.print_exc()
            socketio.emit('goal_result', {
                'success': False,
                'error': str(e)
            })
            socketio.emit('status', {
                'state': 'ready',
                'message': f'Execution error: {str(e)}'
            })
        finally:
            is_running = False
            sys.stdout = original_stdout

    socketio.start_background_task(run_goal)


@socketio.on('stop_session')
def handle_stop_session():
    """Close the WebCane session."""
    global webcane_instance, is_running

    if webcane_instance is None:
        emit('status', {'state': 'connected', 'message': 'No active session.'})
        return

    emit('status', {'state': 'stopping', 'message': 'Closing session...'})
    emit('log', {'message': '━━━ Closing WebCane3 ━━━'})

    def close_session():
        global webcane_instance, is_running
        original_stdout = sys.stdout
        sys.stdout = SocketIOWriter(original_stdout, socketio)

        try:
            webcane_instance.close()
        except Exception as e:
            print(f"[Close] Error: {e}")
        finally:
            webcane_instance = None
            is_running = False
            sys.stdout = original_stdout
            socketio.emit('status', {
                'state': 'connected',
                'message': 'Session closed. Start a new one when ready.'
            })
            socketio.emit('session_stopped', {'success': True})

    socketio.start_background_task(close_session)


if __name__ == '__main__':
    print("=" * 60)
    print("  WebCane3 - Web UI")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
