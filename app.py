from flask import Flask, jsonify, send_file, send_from_directory, request
from flask_cors import CORS
from flask_compress import Compress
import os
import json
import math
from functools import lru_cache

app = Flask(__name__)
CORS(app)
Compress(app)

BASE_DIR = os.path.dirname(__file__)
DATASITE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'DataSite')
CATCHMENTS_DIR = os.path.join(DATASITE_DIR, 'Catchments')
LAYERS_DIR = os.path.join(DATASITE_DIR, 'Layers')
SITEPOLYGON_DIR = os.path.join(DATASITE_DIR, 'SitePolygon')

def safe_path(base_dir, path):
    """Ensure path resolves to a file within base_dir to prevent traversal."""
    abs_path = os.path.abspath(os.path.join(base_dir, path))
    if not abs_path.startswith(os.path.abspath(base_dir)):
        return None
    return abs_path

def slim_feature_collection(geojson):
    if not isinstance(geojson, dict) or 'features' not in geojson:
        return geojson
    slim_features = []
    for f in geojson['features']:
        props = f.get('properties') or {}
        slim_features.append({
            'type': 'Feature',
            'geometry': f.get('geometry'),
            'properties': {
                'name': props.get('name') or props.get('amenity')
                        or props.get('power') or props.get('highway') or ''
            }
        })
    return {'type': 'FeatureCollection', 'features': slim_features}

def filter_by_radius(geojson, lat, lng, radius_km):
    def any_point_within_radius(geometry, lat, lng, radius_km):
        def walk(c):
            if isinstance(c[0], (int, float)):
                flng, flat = c
                dlat = (flat - lat) * 111
                dlng = (flng - lng) * 111 * math.cos(math.radians(lat))
                return math.hypot(dlat, dlng) <= radius_km
            return any(walk(sub) for sub in c)
        try:
            return walk(geometry['coordinates'])
        except Exception:
            return False

    return {
        'type': 'FeatureCollection',
        'features': [
            f for f in geojson.get('features', [])
            if any_point_within_radius(f['geometry'], lat, lng, radius_km)
        ]
    }

@lru_cache(maxsize=32)
def load_layer_cached(filepath, mtime):
    with open(filepath, 'r') as f:
        return json.load(f)

@app.route('/')
def index():
    return send_file(os.path.join(BASE_DIR, 'geospatial_intelligence_module.html'))

@app.route('/Layers/<path:filename>')
def serve_layers(filename):
    filepath = safe_path(LAYERS_DIR, filename)
    if filepath is None or not os.path.exists(filepath):
        return "Not found", 404
    return send_from_directory(os.path.dirname(filepath), os.path.basename(filepath))

@app.route('/SitePolygon/<path:filename>')
def serve_site_polygon(filename):
    filepath = safe_path(SITEPOLYGON_DIR, filename)
    if filepath is None or not os.path.exists(filepath):
        return "Not found", 404
    return send_from_directory(os.path.dirname(filepath), os.path.basename(filepath))

@app.route('/<path:filename>')
def serve_static(filename):
    filepath = safe_path(BASE_DIR, filename)
    if filepath is None or not os.path.exists(filepath):
        return "Not found", 404
    return send_from_directory(os.path.dirname(filepath), os.path.basename(filepath))

@app.route('/Catchments/<path:filename>')
def serve_catchments(filename):
    filepath = safe_path(CATCHMENTS_DIR, filename)
    if filepath is None or not os.path.exists(filepath):
        return "Not found", 404
    return send_from_directory(os.path.dirname(filepath), os.path.basename(filepath))

@app.route('/api/layers', methods=['GET'])
def list_layers():
    if not os.path.exists(CATCHMENTS_DIR):
        return jsonify([])
    
    files = [f for f in os.listdir(CATCHMENTS_DIR) if f.endswith('.fgb')]
    return jsonify(files)

@app.route('/api/layer/<path:filename>', methods=['GET'])
def get_layer(filename):
    if not (filename.endswith('.geojson') or filename.endswith('.json')):
        return jsonify({"error": "Invalid file type"}), 400

    filepath = safe_path(CATCHMENTS_DIR, filename)
    if filepath is None or not os.path.exists(filepath):
        return jsonify({"error": "Not found"}), 404

    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    radius_km = request.args.get('radius_km', type=float)
    
    try:
        mtime = os.path.getmtime(filepath)
        data = load_layer_cached(filepath, mtime)
        data = slim_feature_collection(data)
        
        if lat is not None and lng is not None and radius_km is not None:
            data = filter_by_radius(data, lat, lng, radius_km)
            
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    print("Serving on http://127.0.0.1:5000 via Waitress (threaded)...")
    serve(app, host='127.0.0.1', port=5000, threads=8)
