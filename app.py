import time
import base64
import logging
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

def resize_image_max_dim(img, max_dim=500):
    """Resize image preserving aspect ratio if any dimension exceeds max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def extract_matrix_info(img):
    """Extract matrix dimensions, statistical properties, and a center sample snippet."""
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels == 3 else img
    
    # Statistical intensity metrics
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    mean_val = round(float(np.mean(gray)), 2)
    std_val = round(float(np.std(gray)), 2)
    
    # Color channel means (BGR)
    if channels == 3:
        b_mean = round(float(np.mean(img[:, :, 0])), 2)
        g_mean = round(float(np.mean(img[:, :, 1])), 2)
        r_mean = round(float(np.mean(img[:, :, 2])), 2)
    else:
        b_mean = g_mean = r_mean = mean_val

    # Extract 8x8 sample matrix snippet from image center
    snippet_size = min(8, h, w)
    start_y = max(0, (h - snippet_size) // 2)
    start_x = max(0, (w - snippet_size) // 2)
    snippet = gray[start_y:start_y + snippet_size, start_x:start_x + snippet_size].astype(int).tolist()

    return {
        'shape': [h, w, channels],
        'dtype': str(img.dtype),
        'total_pixels': int(h * w),
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val,
        'channel_means': {'b': b_mean, 'g': g_mean, 'r': r_mean},
        'snippet': snippet,
        'snippet_coords': {'start_x': start_x, 'start_y': start_y, 'size': snippet_size}
    }

def compute_histograms(img):
    """Compute 256-bin pixel intensity distribution histograms using OpenCV cv2.calcHist."""
    channels = img.shape[2] if len(img.shape) == 3 else 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels == 3 else img
    
    # Grayscale / Luminance distribution
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(int).tolist()
    
    # Per-channel distributions (Blue, Green, Red)
    if channels == 3:
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(int).tolist()
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten().astype(int).tolist()
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten().astype(int).tolist()
    else:
        hist_b = hist_g = hist_r = hist_gray

    return {
        'gray': hist_gray,
        'b': hist_b,
        'g': hist_g,
        'r': hist_r
    }

@app.route('/')
def index():
    """Render the main modular Glassmorphic dashboard."""
    return render_template('index.html')

@app.route('/api/match', methods=['POST'])
def match_features():
    """
    Endpoint that takes two uploaded images, extracts keypoints, matches them,
    computes image matrix stats and histograms, and returns results in JSON.
    """
    t_start = time.time()
    
    # 1. Retrieve Parameters
    hessian_threshold = request.form.get('hessian_threshold', default=150, type=float)
    octaves = request.form.get('octaves', default=4, type=int)
    ratio_test = request.form.get('ratio_test', default=0.75, type=float)
    cross_check = request.form.get('cross_check', default='true') == 'true'

    # 2. Retrieve Files
    if 'image_a' not in request.files or 'image_b' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Missing image files in request. Provide both "image_a" and "image_b".'
        }), 400

    file_a = request.files['image_a']
    file_b = request.files['image_b']

    if file_a.filename == '' or file_b.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty file names provided.'
        }), 400

    try:
        # Read files into numpy buffers
        buf_a = np.frombuffer(file_a.read(), np.uint8)
        buf_b = np.frombuffer(file_b.read(), np.uint8)
        
        # Decode into BGR image matrices
        img_a = cv2.imdecode(buf_a, cv2.IMREAD_COLOR)
        img_b = cv2.imdecode(buf_b, cv2.IMREAD_COLOR)
        
        if img_a is None or img_b is None:
            return jsonify({
                'success': False,
                'error': 'Could not decode image files. Ensure files are valid images.'
            }), 400

        # Resize images to match client-side rendering dimensions (max 500px)
        img_a = resize_image_max_dim(img_a, 500)
        img_b = resize_image_max_dim(img_b, 500)

        # Compute Matrix Information & Histograms for both images
        matrix_a = extract_matrix_info(img_a)
        matrix_b = extract_matrix_info(img_b)
        hist_a = compute_histograms(img_a)
        hist_b = compute_histograms(img_b)

        # 3. Create Feature Detector with robust fallback strategies
        method_used = "SURF"
        detector = None

        try:
            # Try initializing SURF
            detector = cv2.xfeatures2d.SURF_create(
                hessianThreshold=hessian_threshold,
                nOctaves=octaves,
                nOctaveLayers=3,
                extended=False,
                upright=False
            )
            logger.info("Successfully initialized OpenCV SURF detector.")
        except (AttributeError, cv2.error) as e:
            logger.warning(f"SURF initialization failed or unavailable ({str(e)}). Falling back to SIFT...")
            try:
                # Fallback to SIFT
                detector = cv2.SIFT_create(
                    nfeatures=0,
                    nOctaveLayers=octaves,
                    contrastThreshold=0.04,
                    edgeThreshold=10,
                    sigma=1.6
                )
                method_used = "SIFT (Fallback)"
                logger.info("Successfully initialized OpenCV SIFT detector.")
            except (AttributeError, cv2.error) as e2:
                logger.warning(f"SIFT initialization failed or unavailable ({str(e2)}). Falling back to ORB...")
                # Fallback to ORB
                detector = cv2.ORB_create(nfeatures=500)
                method_used = "ORB (Fallback)"
                logger.info("Successfully initialized OpenCV ORB detector.")

        # 4. Detect and Compute Descriptors
        kp_a, desc_a = detector.detectAndCompute(img_a, None)
        kp_b, desc_b = detector.detectAndCompute(img_b, None)

        good_matches = []
        kps_a_count = len(kp_a)
        kps_b_count = len(kp_b)

        # 5. Perform Matching using Lowe's Ratio Test
        if desc_a is not None and desc_b is not None and len(desc_a) > 0 and len(desc_b) > 0:
            # Select appropriate Norm based on algorithm type
            if "ORB" in method_used:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

            # Match forward (A -> B)
            k_val = min(2, len(desc_b))
            raw_matches_f = bf.knnMatch(desc_a, desc_b, k=k_val)

            # Apply ratio test
            matches_f = []
            for m_n in raw_matches_f:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < ratio_test * n.distance:
                        matches_f.append(m)
                elif len(m_n) == 1:
                    matches_f.append(m_n[0])

            # Apply Cross-check (bi-directional verification)
            if cross_check:
                k_val_b = min(2, len(desc_a))
                raw_matches_b = bf.knnMatch(desc_b, desc_a, k=k_val_b)
                
                matches_b = {}
                for m_n in raw_matches_b:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < ratio_test * n.distance:
                            matches_b[m.queryIdx] = m.trainIdx
                    elif len(m_n) == 1:
                        matches_b[m_n[0].queryIdx] = m_n[0].trainIdx

                # Intersect forward and backward matches
                for m in matches_f:
                    if m.trainIdx in matches_b and matches_b[m.trainIdx] == m.queryIdx:
                        good_matches.append(m)
            else:
                good_matches = matches_f

        # 6. Draw matches image
        res_img = cv2.drawMatches(
            img_a, kp_a,
            img_b, kp_b,
            good_matches, None,
            matchColor=(0, 212, 6),          # Cyan-emerald color line
            singlePointColor=(139, 92, 246), # Violet color keypoint circles
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # 7. Convert output image to base64
        _, buffer = cv2.imencode('.png', res_img)
        base64_bytes = base64.b64encode(buffer)
        base64_str = base64_bytes.decode('utf-8')
        result_img_url = f"data:image/png;base64,{base64_str}"

        # 8. Compute execution duration
        duration_s = round(time.time() - t_start, 3)

        logger.info(f"Processed match API successfully. Method: {method_used}. Keypoints: A={kps_a_count}, B={kps_b_count}. Matches: {len(good_matches)} in {duration_s}s.")

        return jsonify({
            'success': True,
            'result_image': result_img_url,
            'metrics': {
                'keypoints_a': kps_a_count,
                'keypoints_b': kps_b_count,
                'matches_count': len(good_matches),
                'execution_time_seconds': duration_s,
                'method_used': method_used
            },
            'matrix_info': {
                'image_a': matrix_a,
                'image_b': matrix_b
            },
            'histograms': {
                'image_a': hist_a,
                'image_b': hist_b
            }
        })

    except Exception as e:
        logger.error(f"Error occurred during image processing: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal Server Error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
