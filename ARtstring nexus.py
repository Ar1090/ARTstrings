import cv2
import numpy as np
from flask import Flask, request, render_template_string, send_file
import io
from matplotlib.figure import Figure
import base64

app = Flask(__name__)

HTML_FORM = '''
<!doctype html>
<title>Generador de String Art</title>
<h1>Sube una imagen para generar string art</h1>
<form method=post enctype=multipart/form-data>
  Imagen: <input type=file name=image><br><br>
  Número de clavos: <input type=number name=num_pins value=200><br>
  Radio del círculo (px): <input type=number name=radius value=300><br>
  Número de líneas: <input type=number name=num_lines value=1000><br>
  Diámetro real del círculo (cm): <input type=number step=0.1 name=diameter_cm value=20><br><br>
  <input type=submit value=Generar>
</form>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        num_pins = int(request.form['num_pins'])
        radius = int(request.form['radius'])
        num_lines = int(request.form['num_lines'])
        diameter_cm = float(request.form['diameter_cm'])

        if file:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (600, 600))
            center = (img.shape[1] // 2, img.shape[0] // 2)
            pins = generate_pin_positions(num_pins, radius, center)
            canvas, sequence = draw_string_art(img, pins, num_lines)

            # Draw pin numbers on the canvas
            for idx, (x, y) in enumerate(pins):
                cv2.putText(canvas, str(idx), (x-10, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128), 1)

            fig = Figure()
            ax = fig.subplots()
            ax.imshow(canvas, cmap='gray')
            ax.axis('off')
            output_img = io.BytesIO()
            fig.savefig(output_img, format='png')
            output_img.seek(0)

            spacing = (np.pi * diameter_cm) / num_pins
            spacing_info = f"<p>Separación entre clavos: {spacing:.2f} cm</p>"

            sequence_text = f"Separación entre clavos: {spacing:.2f} cm\n"
            sequence_text += '\n'.join(f"{a} -> {b}" for a, b in sequence)
            output_txt = io.BytesIO()
            output_txt.write(sequence_text.encode('utf-8'))
            output_txt.seek(0)

            with open("sequence.txt", "w") as f:
                f.write(sequence_text)

            return f'''
            <h2>Resultado:</h2>
            <img src="data:image/png;base64,{encode_image(output_img)}"><br><br>
            {spacing_info}
            <a href="/download_image">Descargar imagen</a><br>
            <a href="/download_sequence">Descargar secuencia</a><br>
            <a href="/">Volver</a>
            '''

    return render_template_string(HTML_FORM)

def encode_image(output_img):
    return base64.b64encode(output_img.read()).decode('utf-8')

@app.route('/download_image')
def download_image():
    return send_file("preview.png", mimetype='image/png', as_attachment=True, download_name='preview.png')

@app.route('/download_sequence')
def download_sequence():
    return send_file("sequence.txt", mimetype='text/plain', as_attachment=True, download_name='sequence.txt')

def generate_pin_positions(n_pins, radius, center):
    return [
        (
            int(center[0] + radius * np.cos(2 * np.pi * i / n_pins)),
            int(center[1] + radius * np.sin(2 * np.pi * i / n_pins))
        )
        for i in range(n_pins)
    ]

def draw_string_art(image, pins, n_lines):
    canvas = np.ones_like(image) * 255
    gray_img = image.copy()
    line_sequence = []
    current_pin = 0

    for _ in range(n_lines):
        best_pin = None
        best_score = -np.inf

        for i, pin in enumerate(pins):
            if i == current_pin:
                continue
            mask = np.zeros_like(image)
            cv2.line(mask, pins[current_pin], pin, 255, 1)
            score = np.sum((255 - gray_img) * (mask / 255))
            if score > best_score:
                best_score = score
                best_pin = i

        if best_pin is not None:
            cv2.line(canvas, pins[current_pin], pins[best_pin], 0, 1)
            cv2.line(gray_img, pins[current_pin], pins[best_pin], 255, 1)
            line_sequence.append((current_pin, best_pin))
            current_pin = best_pin

    cv2.imwrite("preview.png", canvas)

    return canvas, line_sequence

if __name__ == '__main__':
    app.run(debug=True)
