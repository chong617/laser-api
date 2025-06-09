from flask import Flask, request, jsonify
from flask_cors import CORS
from laserembeddings import Laser

app = Flask(__name__)
CORS(app)

laser = Laser()

@app.route("/embed", methods=["POST"])
def embed_text():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is empty"}), 400
    try:
        # laser.embed_sentences expects a list of sentences
        embedding = laser.embed_sentences([text], lang='en')[0]
        return jsonify({"embedding": embedding.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000)

