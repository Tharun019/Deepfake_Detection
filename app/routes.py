import os

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from modules.binary.analyzer import analyze as analyze_binary
from modules.content.analyzer import analyze as analyze_content
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

    filename = secure_filename(uploaded.filename)
    save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    uploaded.save(save_path)

    try:
        metadata_score = analyze_metadata(save_path, media_type)
        content_score = analyze_content(save_path, media_type)
        binary_score = analyze_binary(save_path, media_type)
        result = fuse_scores(metadata_score, content_score, binary_score)
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

    return jsonify(result)
