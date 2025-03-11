from flask import Flask, request, jsonify
import mysql.connector
import os
import cv2
import numpy as np

app = Flask(__name__)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Pdpg050399",
    database="mr_robot"
)
cursor = db.cursor()

UPLOAD_FOLDER = "images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Error: No se pudo cargar una de las imágenes.")
        return 0  # O algún valor que indique error

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Verificar si se encontraron descriptores en ambas imágenes
    if des1 is None or des2 is None:
        print("Error: No se encontraron suficientes características en una o ambas imágenes.")
        return 0

    # Asegurar que los descriptores tienen el mismo tipo
    des1 = des1.astype('float32')
    des2 = des2.astype('float32')

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    return len(matches)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el archivo"}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    cursor.execute("SELECT id, image_path FROM students")
    students = cursor.fetchall()

    for student in students:
        student_id, image_path = student
        similarity = compare_images(file_path, image_path)
        if similarity > 200:
            return jsonify({"mensaje": f"Bienvenido, student {student_id}!"})

    cursor.execute("INSERT INTO students (image_path) VALUES (%s)", (file_path,))
    db.commit()

    return jsonify({"mensaje": "Estudiante no reconocido, imagen guardada"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)