"""
gee_backend.py  —  Flask + Earth Engine Python API backend
============================================================
Exposes two endpoints for the HTML dashboard:

  POST /api/analyze
    Body: { startDate, endDate, intervalMonths }
    Returns: per-site metrics for every generated period
             + map tile URL templates for RGB & Classification layers

  GET /api/status
    Returns { authenticated: true/false } for the frontend to know
    whether to show the "please set up service account" banner.

SETUP
-----
1. pip install flask flask-cors earthengine-api
2. Authenticate once:
     earthengine authenticate          (for personal dev)
   OR create a Service Account key JSON and point to it:
     export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
3. python gee_backend.py
   Server runs at http://localhost:5050

NOTE: The heavy EE calls (composite building, training, classification)
can take 60-120 s for all 5 sites × many periods. The frontend shows a
live progress banner via Server-Sent Events on GET /api/analyze/stream.
"""

import os, json, threading, queue, datetime
from math import floor
from dateutil.relativedelta import relativedelta   # pip install python-dateutil
import ee
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
GEE_PROJECT    = 'vital-plating-495019-n7'
TRAINING_ASSET = f'projects/{GEE_PROJECT}/assets/HorizonIP_Training_Points_14P'
POLY_ASSET     = f'projects/{GEE_PROJECT}/assets/TrainingPolygons_geometry_geometry2_geometry3'

SITES = [
    {'site_id': 'SITE01', 'site_name': 'AlphaOne',  'asset': f'projects/{GEE_PROJECT}/assets/AlphaOne'},
    {'site_id': 'SITE02', 'site_name': 'Ascendes',  'asset': f'projects/{GEE_PROJECT}/assets/Ascendes'},
    {'site_id': 'SITE03', 'site_name': 'ESR',       'asset': f'projects/{GEE_PROJECT}/assets/ESR'},
    {'site_id': 'SITE04', 'site_name': 'EndoSpace', 'asset': f'projects/{GEE_PROJECT}/assets/EndoSpace'},
    {'site_id': 'SITE05', 'site_name': 'Vengudi',   'asset': f'projects/{GEE_PROJECT}/assets/Vengudi'},
]

S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
ML_FEATURES = [
    'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
    'NDVI', 'NDBI', 'MNDWI', 'BSI', 'EVI', 'IBI', 'NBI',
    'BRIGHT', 'WHITENESS', 'FLATNESS',
    'R_B4_B11', 'R_B8_B11', 'R_B3_B4',
    'TEX_CONTRAST', 'TEX_CORR', 'TEX_ENERGY'
]

RGB_VIZ   = {'bands': ['B4', 'B3', 'B2'], 'min': 0.03, 'max': 0.28, 'gamma': 1.4}
CLASS_VIZ = {'min': 1, 'max': 3, 'palette': ['#4CAF50', '#FF6600', '#1565C0']}

# ── EE INIT ────────────────────────────────────────────────────────────────────
_ee_ready = False

def init_ee():
    global _ee_ready
    try:
        ee.Initialize(project=GEE_PROJECT)
        _ee_ready = True
        print("[GEE] Earth Engine initialized successfully.")
    except Exception as ex:
        print(f"[GEE] Could not initialize Earth Engine: {ex}")
        _ee_ready = False

init_ee()

# ── FLASK APP ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow requests from the HTML file opened locally

# ── HELPERS ────────────────────────────────────────────────────────────────────
def generate_periods(start_str, end_str, interval_months):
    """Return a list of period dicts with id, label, start, end, desc."""
    periods = []
    start   = datetime.date.fromisoformat(start_str)
    end     = datetime.date.fromisoformat(end_str)
    curr    = start
    idx     = 1
    while curr < end:
        p_start = curr
        p_end   = curr + relativedelta(months=interval_months) - datetime.timedelta(days=1)
        if p_end > end:
            p_end = end
        pid   = f'T{idx:02d}'
        label = (f"{pid}_{p_start.strftime('%b')}-{p_end.strftime('%b')}_{p_start.year}")
        desc  = f"{p_start.isoformat()} to {p_end.isoformat()}"
        periods.append({'id': pid, 'label': label,
                        'start': p_start.isoformat(), 'end': p_end.isoformat(),
                        'desc': desc})
        curr += relativedelta(months=interval_months)
        idx  += 1
    return periods


def mask_s2(img):
    qa   = img.select('QA60')
    mask = (qa.bitwiseAnd(1 << 10).eq(0)
              .And(qa.bitwiseAnd(1 << 11).eq(0)))
    return (img.updateMask(mask).divide(10000)
              .select(S2_BANDS)
              .copyProperties(img, ['system:time_start']))


def get_composite(p, aoi):
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(aoi)
             .filterDate(p['start'], p['end'])
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
             .map(mask_s2))
    col_relax = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterBounds(aoi)
                   .filterDate(p['start'], p['end'])
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
                   .map(mask_s2))
    final_col = ee.ImageCollection(
        ee.Algorithms.If(col.size().gt(0), col, col_relax))
    dummy = ee.Image.constant([0]*6).rename(S2_BANDS).clip(aoi)
    return ee.Image(
        ee.Algorithms.If(final_col.size().gt(0), final_col.median().clip(aoi), dummy))


def build_stack(img, aoi):
    b2,b3,b4 = img.select('B2'), img.select('B3'), img.select('B4')
    b8,b11,b12 = img.select('B8'), img.select('B11'), img.select('B12')

    ndvi  = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndbi  = img.normalizedDifference(['B11', 'B8']).rename('NDBI')
    mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    bsi   = img.expression(
        '((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))',
        {'B11': b11, 'B4': b4, 'B8': b8, 'B2': b2}).rename('BSI')
    evi   = img.expression(
        '2.5*((B8-B4)/(B8+6*B4-7.5*B2+1))',
        {'B8': b8, 'B4': b4, 'B2': b2}).rename('EVI')
    ibi   = img.expression(
        '(2*B11/(B11+B8)-(B8/(B8+B4)+B3/(B3+B11)))/(2*B11/(B11+B8)+(B8/(B8+B4)+B3/(B3+B11)))',
        {'B11': b11, 'B8': b8, 'B4': b4, 'B3': b3}).rename('IBI')
    nbi   = img.normalizedDifference(['B11', 'B4']).rename('NBI')

    bright    = b2.add(b3).add(b4).divide(3).rename('BRIGHT')
    whiteness = b2.add(b3).add(b4).subtract(b8.add(b11).add(b12)).abs().multiply(-1).rename('WHITENESS')
    flatness  = b11.divide(b4.add(0.0001)).rename('FLATNESS')
    r_b4_b11  = b4.divide(b11.add(0.001)).rename('R_B4_B11')
    r_b8_b11  = b8.divide(b11.add(0.001)).rename('R_B8_B11')
    r_b3_b4   = b3.divide(b4.add(0.001)).rename('R_B3_B4')

    glcm  = b4.multiply(10000).toInt32().glcmTexture(size=3, kernel=ee.Kernel.square(1))
    tex_c = glcm.select('B4_contrast').rename('TEX_CONTRAST')
    tex_r = glcm.select('B4_corr').rename('TEX_CORR')
    tex_e = glcm.select('B4_asm').rename('TEX_ENERGY')

    return img.addBands([
        ndvi, ndbi, mndwi, bsi, evi, ibi, nbi,
        bright, whiteness, flatness, r_b4_b11, r_b8_b11, r_b3_b4,
        tex_c, tex_r, tex_e
    ]).clip(aoi)


def compute_progress_pct(classified, aoi):
    """Returns a dict with server-side EE objects (not yet evaluated)."""
    valid_mask   = classified.neq(4)
    dev_only     = classified.updateMask(valid_mask)
    area_stats   = dev_only.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=aoi, scale=20, maxPixels=1e7)
    counts = ee.Dictionary(area_stats.get('stage'))
    c1 = ee.Number(counts.get('1', 0))
    c2 = ee.Number(counts.get('2', 0))
    c3 = ee.Number(counts.get('3', 0))
    total = c1.add(c2).add(c3).max(1)
    return {
        'notStartedPct':       c1.divide(total).multiply(100).round(),
        'underConstructionPct': c2.divide(total).multiply(100).round(),
        'completedPct':        c3.divide(total).multiply(100).round(),
    }


def sample_polygon_across_sites(geom, label, site_stack_data, period_idx):
    per_site = []
    for sd in site_stack_data:
        site_geom = ee.Geometry(geom).intersection(sd['aoi'], ee.ErrorMargin(1))
        has_area  = site_geom.area(1).gt(1)
        stack     = sd['stacks'][period_idx]
        samples   = ee.FeatureCollection(
            ee.Algorithms.If(
                has_area,
                stack.select(ML_FEATURES).sampleRegions(
                    collection=ee.FeatureCollection([ee.Feature(site_geom)]),
                    properties=[], scale=10, tileScale=4, geometries=True),
                ee.FeatureCollection([])))
        per_site.append(samples.map(lambda f: f.set('label', label)))
    return ee.FeatureCollection(per_site).flatten()


# ── CORE ANALYSIS FUNCTION (runs in a background thread) ─────────────────────
def run_analysis_pipeline(start_str, end_str, interval_months, progress_q):
    """
    Runs the full GEE pipeline and puts results into progress_q.
    progress_q receives dicts: { type: 'progress'|'result'|'error', ... }
    """
    try:
        def emit(msg):
            progress_q.put({'type': 'progress', 'message': msg})

        emit('Generating time periods...')
        periods = generate_periods(start_str, end_str, interval_months)
        if not periods:
            raise ValueError('No periods generated — check your date range and interval.')
        latest_idx = len(periods) - 1
        past_idx   = floor(len(periods) / 2)

        emit(f'Generated {len(periods)} periods. Loading training polygons...')
        poly_fc   = ee.FeatureCollection(POLY_ASSET)
        geom_co   = poly_fc.filter(ee.Filter.eq('poly_name', 'completed')).geometry()
        geom_uc   = poly_fc.filter(ee.Filter.eq('poly_name', 'under_construction')).geometry()
        geom_ns   = poly_fc.filter(ee.Filter.eq('poly_name', 'not_started')).geometry()

        emit('Building per-site Sentinel-2 stacks (this may take a while)...')
        site_stack_data = []
        for site in SITES:
            aoi        = ee.FeatureCollection(site['asset']).geometry()
            composites = [get_composite(p, aoi) for p in periods]
            stacks     = [build_stack(c, aoi) for c in composites]
            site_stack_data.append({
                'site': site, 'aoi': aoi,
                'composites': composites, 'stacks': stacks
            })

        emit('Loading shared training points...')
        shared_pts = (ee.FeatureCollection(TRAINING_ASSET)
                        .filter(ee.Filter.neq('label', 4)))

        emit('Extracting polygon training samples...')
        co_samples = sample_polygon_across_sites(geom_co, 3, site_stack_data, latest_idx)
        uc_samples = sample_polygon_across_sites(geom_uc, 2, site_stack_data, latest_idx)
        ns_samples = sample_polygon_across_sites(geom_ns, 1, site_stack_data, past_idx)
        all_pts    = shared_pts.merge(co_samples).merge(uc_samples).merge(ns_samples)

        emit('Training Random Forest classifier (250 trees)...')
        classifier = (ee.Classifier.smileRandomForest(
                          numberOfTrees=250, variablesPerSplit=5,
                          minLeafPopulation=3, bagFraction=0.75, seed=42)
                      .train(features=all_pts,
                             classProperty='label',
                             inputProperties=ML_FEATURES))

        emit('Classifying all sites across all periods...')
        site_results = []
        for sd in site_stack_data:
            classified = [
                s.select(ML_FEATURES).classify(classifier)
                  .rename('stage').clip(sd['aoi']).toByte()
                for s in sd['stacks']
            ]
            metrics_list = [compute_progress_pct(c, sd['aoi']) for c in classified]
            site_results.append({
                'site': sd['site'], 'aoi': sd['aoi'],
                'stacks': sd['stacks'],
                'classified': classified,
                'metrics_list': metrics_list
            })

        emit('Packaging server-side objects for evaluation...')
        # Build one big nested EE structure so we only need ONE .getInfo() call
        ee_output = ee.List([
            ee.Dictionary({
                'site_id': res['site']['site_id'],
                'site_name': res['site']['site_name'],
                'metrics': ee.List([
                    ee.Dictionary({
                        'notStartedPct':        m['notStartedPct'],
                        'underConstructionPct': m['underConstructionPct'],
                        'completedPct':         m['completedPct'],
                    })
                    for m in res['metrics_list']
                ])
            })
            for res in site_results
        ])

        emit('Evaluating EE computations on Google servers (60-120 s)...')
        client_metrics = ee_output.getInfo()   # ← blocking call

        emit('Fetching map tile URLs for the latest period...')
        tiles = {}
        for res in site_results:
            sid = res['site']['site_id']
            rgb_vis   = res['stacks'][latest_idx].visualize(**RGB_VIZ)
            class_vis = res['classified'][latest_idx].visualize(**CLASS_VIZ)
            rgb_url   = rgb_vis.getMapId()['tile_fetcher'].url_format
            class_url = class_vis.getMapId()['tile_fetcher'].url_format
            tiles[sid] = {'rgbUrl': rgb_url, 'classUrl': class_url}

        progress_q.put({
            'type': 'result',
            'periods': periods,
            'metrics': client_metrics,
            'tiles':   tiles,
        })

    except Exception as ex:
        progress_q.put({'type': 'error', 'message': str(ex)})

# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route('/api/status')
def status():
    return jsonify({'authenticated': _ee_ready, 'project': GEE_PROJECT})

@app.route('/api/analyze/stream')
def analyze_stream():
    """
    SSE endpoint.  Frontend calls:
      const src = new EventSource('/api/analyze/stream?startDate=...&endDate=...&intervalMonths=...')
    and receives progress messages followed by the final result JSON.
    """
    start_str       = request.args.get('startDate', '2021-07-01')
    end_str         = request.args.get('endDate',   '2026-06-30')
    interval_months = int(request.args.get('intervalMonths', 6))

    if not _ee_ready:
        def error_gen():
            yield "data: " + json.dumps({'type': 'error',
                'message': 'Earth Engine is not authenticated on the server.'}) + "\n\n"
        return Response(stream_with_context(error_gen()),
                        mimetype='text/event-stream')

    q = queue.Queue()
    thread = threading.Thread(
        target=run_analysis_pipeline,
        args=(start_str, end_str, interval_months, q),
        daemon=True
    )
    thread.start()

    def generate():
        while True:
            item = q.get()
            yield "data: " + json.dumps(item) + "\n\n"
            if item['type'] in ('result', 'error'):
                break

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control':    'no-cache',
                        'X-Accel-Buffering': 'no'
                    })


@app.route('/api/analyze', methods=['POST'])
def analyze_sync():
    """
    Synchronous fallback — blocks until analysis is complete.
    Use the SSE endpoint above for live progress updates.
    """
    if not _ee_ready:
        return jsonify({'error': 'Earth Engine not authenticated.'}), 503

    body            = request.get_json(force=True)
    start_str       = body.get('startDate', '2021-07-01')
    end_str         = body.get('endDate',   '2026-06-30')
    interval_months = int(body.get('intervalMonths', 6))

    q = queue.Queue()
    run_analysis_pipeline(start_str, end_str, interval_months, q)

    while True:
        item = q.get()
        if item['type'] == 'result':
            return jsonify(item)
        elif item['type'] == 'error':
            return jsonify({'error': item['message']}), 500


if __name__ == '__main__':
    print("Starting GEE Backend on http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
