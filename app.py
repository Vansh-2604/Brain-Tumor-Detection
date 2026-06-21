from flask import Flask, render_template, request, redirect, url_for, session

import os

from predict import predict_image

app = Flask(__name__)
app.secret_key="brain_tumor_secret"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():

    prediction = session.pop("prediction", None)
    confidence = session.pop("confidence", None)
    confidence_width = session.pop("confidence_width", None)
    image_path = session.pop("image_path", None)

    return render_template(
        "index.html",
        prediction = prediction,
        confidence = confidence,
        confidence_width=confidence_width,
        image_path = image_path

        )

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    prediction, confidence = predict_image(
        filepath
    )

    if confidence>=95:
        confidence_width=100
    else:
        confidence_width = round(confidence)
    image_path = "uploads/" + file.filename

    session["prediction"] = prediction
    session["confidence"] = confidence
    session["confidence_width"] = confidence_width
    session["image_path"] = image_path

    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)