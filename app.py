"""
app.py

Flask backend for the template matching demo.
Serves the HTML page and handles the /match endpoint that runs detection.
"""

import io
import base64

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from template_matcher import pyramid_match


app = Flask(__name__)
CORS(app)


def decode_image_from_upload(file_storage):
    """
    Convert an uploaded file (Flask's FileStorage object) into a BGR numpy image.
    """
    # Read the raw bytes
    raw_bytes = file_storage.read()
    # Convert to a numpy array of uint8s
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    # Decode as a color image (cv2 will return BGR)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def encode_image_as_base64(bgr_image):
    """
    Encode a BGR numpy image as a base64 PNG data URL the browser can display.
    """
    success, buffer = cv2.imencode(".png", bgr_image)
    if not success:
        raise RuntimeError("Failed to encode image as PNG")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/match", methods=["POST"])
def match():
    """
    Receive a main image and a template image, run pyramid_match,
    draw the bounding box, and return the result + the score.
    """
    # Validate uploads
    if "main_image" not in request.files or "template_image" not in request.files:
        return jsonify({"error": "Both main_image and template_image are required"}), 400

    # Decode both uploads as color BGR images
    main_bgr = decode_image_from_upload(request.files["main_image"])
    template_bgr = decode_image_from_upload(request.files["template_image"])

    if main_bgr is None or template_bgr is None:
        return jsonify({"error": "Could not decode one of the images"}), 400

    # Convert both to grayscale for matching
    main_gray = cv2.cvtColor(main_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # Sanity check: template must fit inside main image
    if (template_gray.shape[0] > main_gray.shape[0]
            or template_gray.shape[1] > main_gray.shape[1]):
        return jsonify({
            "error": (
                f"Template ({template_gray.shape}) must be smaller than "
                f"the main image ({main_gray.shape})"
            )
        }), 400

    # Run pyramid match
    try:
        y, x, score = pyramid_match(main_gray, template_gray, num_levels=3)
    except Exception as e:
        return jsonify({"error": f"Matching failed: {e}"}), 500

    # Draw the bounding box on a copy of the main image (BGR)
    output = main_bgr.copy()
    tmpl_h, tmpl_w = template_gray.shape
    cv2.rectangle(
        output,
        (int(x), int(y)),
        (int(x + tmpl_w), int(y + tmpl_h)),
        color=(0, 255, 0),  # green in BGR
        thickness=4,
    )

    # Encode the output image as base64 to send back to the browser
    result_data_url = encode_image_as_base64(output)

    return jsonify({
        "score": float(score),
        "x": int(x),
        "y": int(y),
        "template_width": int(tmpl_w),
        "template_height": int(tmpl_h),
        "result_image": result_data_url,
    })


if __name__ == "__main__":
    # debug=True gives auto-reload on code changes and detailed error pages
    app.run(host="127.0.0.1", port=5000, debug=True)