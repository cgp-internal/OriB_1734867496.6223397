import os
from flask import Flask, request, send_file, jsonify, render_template, jsonify, send_file, make_response
from helpers import _generate, _detect


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        id = int(data.get("id"))
        if not isinstance(id, int) or not (0 <= id <= 50):
            return (
                jsonify(
                    {"error": "Invalid 'id', must be an integer between 0 and 50."}
                ),
                400,
            )

        size = int(data.get("size", 400))
        border = int(data.get("border", 50))

        _generate(id, size, border, output_filename="generated_aruco.png")
        return send_file("generated_aruco.png", as_attachment=True)

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


import base64

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        image = request.files["image"]
        filename = image.filename
        file_path = filename
        output_file = "detected_aruco.png"

        image.save(file_path)

        # Call the _detect function to process the image
        result = _detect(file_path)

        if not result:
            return jsonify({"error": "No markers detected."}), 404

        # Encode the processed image as base64
        with open(output_file, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({"data": result, "image": image_base64}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


# @app.route("/detect", methods=["POST"])
# def detect():
#     try:
#         if "image" not in request.files:
#             return jsonify({"error": "No image file provided."}), 400

#         image = request.files["image"]
#         filename = image.filename 
#         # file_path = os.path.join("uploads", filename)  # Save to 'uploads' folder
#         file_path = filename

#         image.save(file_path) 

#         result = _detect(file_path)
#         return jsonify(result)

#     except Exception as e:        
#         print(e)
#         return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
