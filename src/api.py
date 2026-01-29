from flask import Flask, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='../static')
CORS(app)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(f'{app.static_folder}/css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(f'{app.static_folder}/js', filename)

if __name__ == '__main__':
    print("🚀 Server running at http://localhost:5000")
    app.run(debug=True, port=5000)