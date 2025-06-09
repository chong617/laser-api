from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# 使用轻量模型，内存友好
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/embed", methods=["POST"])
def embed_text():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is empty"}), 400
    try:
        embedding = model.encode([text])[0]
        return jsonify({"embedding": embedding.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
