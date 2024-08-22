from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory
from app.services.faiss_service import FAISSService
from app.models.models import FrameMetadata
import os
import csv
import json

main = Blueprint('main', __name__)

faiss_index = None
selected_frames = set()


@main.before_app_first_request
def initialize_faiss():
    global faiss_index, selected_frames
    selected_frames = set()
    faiss_index = FAISSService()


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/search', methods=['POST', 'GET'])
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        metadata_query = request.form.get('metadata_query')
    else:
        query = request.args.get('query')
        metadata_query = request.args.get('metadata_query')

    page = request.args.get('page', 1, type=int)
    per_page = 15

    if metadata_query:
        try:
            metadata_query = json.loads(metadata_query)
            results = faiss_index.search(metadata_query)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid metadata query format"}), 400
    else:
        results = faiss_index.search(query)

    total_results = len(results)
    total_pages = (total_results + per_page - 1) // per_page
    paginated_results = paginate(results, page, per_page)

    return render_template('search_results.html', results=paginated_results, query=query, page=page, total_pages=total_pages)


@main.route('/frame/<frame_id>')
def frame_details(frame_id):
    metadata = FrameMetadata.get(frame_id)
    return jsonify(metadata)


@main.route('/selected-frames')
def get_selected_frames():
    page = request.args.get('page', 1, type=int)
    per_page = 20

    frames = [FrameMetadata.get(frame_id) for frame_id in selected_frames]
    total_frames = len(frames)
    total_pages = (total_frames + per_page - 1) // per_page
    paginated_frames = paginate(frames, page, per_page)

    return render_template('selected_frames.html', frames=paginated_frames, page=page, total_pages=total_pages)


@main.route('/toggle-frame', methods=['POST'])
def toggle_frame():
    frame_id = request.form.get('frame_id')
    if frame_id in selected_frames:
        selected_frames.remove(frame_id)
    else:
        selected_frames.add(frame_id)
    return get_selected_frames()


@main.route('/submit-frames', methods=['POST'])
def submit_frames():
    frame_ids = request.form.getlist('frame_ids')
    with open(current_app.config['RESULTS_CSV_PATH'], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for frame_id in frame_ids:
            metadata = FrameMetadata.get(frame_id)
            writer.writerow(
                [frame_id, metadata['video_path'], metadata['timestamp']])
    return "Results saved successfully"


@main.route('/video-preview/<frame_id>')
def video_preview(frame_id):
    metadata = FrameMetadata.get(frame_id)
    video_path = metadata['video_path']
    timestamp = metadata['timestamp']
    return render_template('video_preview.html', video_path=video_path, timestamp=timestamp)


@main.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(current_app.config['VIDEOS_DIR'], filename)


@main.route('/keyframes/<path:filename>')
def serve_keyframe(filename):
    return send_from_directory(current_app.config['KEYFRAMES_DIR'], filename)


def paginate(items, page, per_page):
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end]
