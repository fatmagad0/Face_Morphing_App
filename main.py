import cv2
import numpy as np
import dlib
import imageio
import os
import uuid
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder='static', template_folder='templates')
UPLOAD_FOLDER = os.path.join('static', 'outputs')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        raise ValueError("No face detected in the image!")
    shape = predictor(gray, rects[0])
    points = []
    for i in range(68):
        points.append((shape.part(i).x, shape.part(i).y))
    h, w = image.shape[:2]
    border_points = [(0,0), (w//2, 0), (w-1, 0), (w-1, h//2), (w-1, h-1), (w//2, h-1), (0, h-1), (0, h//2)]
    points.extend(border_points)
    return np.array(points, np.float32)

def get_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    point_to_idx = {}
    for i, p in enumerate(points):
        pt = (int(p[0]), int(p[1]))
        subdiv.insert(pt)
        point_to_idx[pt] = i
    triangle_list = subdiv.getTriangleList()
    delaunay_triangles = []
    for t in triangle_list:
        pt1, pt2, pt3 = (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))
        if pt1 in point_to_idx and pt2 in point_to_idx and pt3 in point_to_idx:
            delaunay_triangles.append((point_to_idx[pt1], point_to_idx[pt2], point_to_idx[pt3]))
    return delaunay_triangles

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    t1_rect = [(pt[0]-r1[0], pt[1]-r1[1]) for pt in t1]
    t2_rect = [(pt[0]-r2[0], pt[1]-r2[1]) for pt in t2]
    t_rect = [(pt[0]-r[0], pt[1]-r[1]) for pt in t]
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)
    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2_rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    warpImage1 = apply_affine_transform(img1_rect, t1_rect, t_rect, (r[2], r[3]))
    warpImage2 = apply_affine_transform(img2_rect, t2_rect, t_rect, (r[2], r[3]))
    img_rect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask

def generate_morph(img1, img2, points1, points2, alpha):
    img1, img2 = np.float32(img1), np.float32(img2)
    points_m = (1 - alpha) * points1 + alpha * points2
    h, w = img1.shape[:2]
    rect = (0, 0, w, h)
    triangles = get_delaunay_triangles(rect, points_m)
    img_morph = np.zeros(img1.shape, dtype=img1.dtype)
    for t in triangles:
        pt1 = [points1[t[0]], points1[t[1]], points1[t[2]]]
        pt2 = [points2[t[0]], points2[t[1]], points2[t[2]]]
        pt_m = [points_m[t[0]], points_m[t[1]], points_m[t[2]]]
        morph_triangle(img1, img2, img_morph, pt1, pt2, pt_m, alpha)
    return np.uint8(img_morph)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/morph', methods=['POST'])
def morph():
    try:
        f1, f2 = request.files['face1'], request.files['face2']
        unique_id = uuid.uuid4().hex
        p1, p2 = f"temp1_{unique_id}.jpg", f"temp2_{unique_id}.jpg"
        f1.save(p1); f2.save(p2)

        img1, img2 = cv2.imread(p1), cv2.imread(p2)
        # h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
        # img1, img2 = cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))

        MAX_SIZE = 500 
        h = min(img1.shape[0], img2.shape[0], MAX_SIZE)
        w = min(img1.shape[1], img2.shape[1], MAX_SIZE)
        
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

        pts1, pts2 = get_landmarks(img1), get_landmarks(img2)
        frames = []
        for i in range(0, 15):
            alpha = i / 14.0
            frame = generate_morph(img1, img2, pts1, pts2, alpha)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last_frame = frames[-1]
        for _ in range(15):
            frames.append(last_frame)
        output_filename = f"result_{unique_id}.gif"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        imageio.mimsave(output_path, frames, fps=10,loop=0)
        
        os.remove(p1); os.remove(p2)
        return jsonify({"gif_url": f"/static/outputs/{output_filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)