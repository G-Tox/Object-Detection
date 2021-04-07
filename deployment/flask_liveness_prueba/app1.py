from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index1.html')

@socketio.on('message')
def handle_message(message1):
    print('received message: ' + message1)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0')