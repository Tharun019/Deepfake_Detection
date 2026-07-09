import os
from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename
from modules.binary.analyzer import analyze as analyze_binary
from modules.content.analyzer import analyze as analyze_content, generate_gradcam
from modules.metadata.analyzer import analyze as analyze_metadata
from modules.scoring.fusion import fuse_scores

main = Blueprint('main', __name__)
ALLOWED_TABS = {'image', 'video', 'audio'}

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    uploaded = request.files['file']
    if not uploaded or not uploaded.filename:
        return jsonify({'error': 'No file selected'}), 400

    media_type = request.form.get('currentTab', 'image')
    if media_type not in ALLOWED_TABS:
        return jsonify({'error': 'Invalid media type'}), 400

    run_metadata = request.form.get('layer_metadata', '1') == '1'
    run_content  = request.form.get('layer_content',  '1') == '1'
    run_binary   = request.form.get('layer_binary',   '1') == '1'
    run_xai      = request.form.get('layer_xai',      '1') == '1'

    filename  = secure_filename(uploaded.filename)
    save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    uploaded.save(save_path)

    try:
        # L1 — returns (score, features) tuple
        if run_metadata:
            metadata_result = analyze_metadata(save_path, media_type)
            metadata_score    = metadata_result[0]
            metadata_features = metadata_result[1]
        else:
            metadata_score    = 0.5
            metadata_features = {}

        # L2 — returns float only
        content_score = analyze_content(save_path, media_type) if run_content else 0.5

        # L3 — returns (score, features) tuple
        if run_binary:
            binary_result = analyze_binary(save_path, media_type)
            binary_score    = binary_result[0]
            binary_features = binary_result[1]
        else:
            binary_score    = 0.5
            binary_features = {}

        # Fusion receives plain floats — untouched
        result = fuse_scores(metadata_score, content_score, binary_score)

        # Attach feature breakdowns to response
        result['xai'] = {
            'metadata_features': metadata_features,
            'binary_features':   binary_features
        }

        # Grad-CAM — image only, XAI + content toggles must be on
        if media_type == 'image' and run_xai and run_content:
            gradcam = generate_gradcam(save_path)
            if gradcam:
                result['gradcam_b64'] = gradcam

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    return jsonify(result)