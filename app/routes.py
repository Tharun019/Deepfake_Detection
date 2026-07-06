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

    # Read layer toggle flags sent from the frontend (default enabled)
    run_metadata = request.form.get('layer_metadata', '1') == '1'
    run_content  = request.form.get('layer_content',  '1') == '1'
    run_binary   = request.form.get('layer_binary',   '1') == '1'
    run_xai      = request.form.get('layer_xai',      '1') == '1'

    filename  = secure_filename(uploaded.filename)
    save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    uploaded.save(save_path)

    try:
        # Only run layers that are toggled on; use neutral 0.5 for disabled ones
        metadata_score = analyze_metadata(save_path, media_type) if run_metadata else 0.5
        content_score  = analyze_content(save_path, media_type)  if run_content  else 0.5
        binary_score   = analyze_binary(save_path, media_type)   if run_binary   else 0.5

        result = fuse_scores(metadata_score, content_score, binary_score)

        # Grad-CAM: images only, requires content layer + XAI toggle enabled
        if media_type == 'image' and run_xai and run_content:
            gradcam = generate_gradcam(save_path)
            if gradcam:
                result['gradcam_b64'] = gradcam

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    return jsonify(result)
