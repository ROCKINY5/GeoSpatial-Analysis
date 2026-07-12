from flask import Flask, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
DATASITE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'DataSite')
CATCHMENTS_DIR = os.path.join(DATASITE_DIR, 'Catchments')
LAYERS_DIR = os.path.join(DATASITE_DIR, 'Layers')

@app.route('/')
def index():
    return send_file(os.path.join(BASE_DIR, 'geospatial_intelligence_module.html'))

@app.route('/Layers/<path:filename>')
def serve_layers(filename):
    return send_from_directory(LAYERS_DIR, filename)

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)

@app.route('/api/layers', methods=['GET'])
def list_layers():
    if not os.path.exists(CATCHMENTS_DIR):
        return jsonify([])
    
    files = [f for f in os.listdir(CATCHMENTS_DIR) if f.endswith('.geojson') or f.endswith('.json')]
    return jsonify(files)

@app.route('/api/layer/<path:filename>', methods=['GET'])
def get_layer(filename):
    filepath = os.path.join(CATCHMENTS_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Not found"}), 404
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
