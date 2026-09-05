"""
Flask front end for TestGenerationAgent.

Routes
------
GET  /            -> the two-column workbench page
POST /generate     -> body: {concept, questions: [...], answers: [...]}
                      calls agent.generate_example(concept, few_shot_examples)
                      returns {id, question, answer, attempts, accepted}
POST /answer        -> body: {feedback, user_answer}
                      calls agent.record_feedback(doc_id, feedback, user_answer)
                      for whichever example /generate last produced
                      returns {status: "ok"}
POST /visual        -> calls agent.plot_summary(...), saves a PNG under
                      static/, returns {url: "/static/plot_<ts>.png"}

Run:
    pip install flask langchain-google-genai langchain-chroma langchain-core chromadb matplotlib
    export GOOGLE_API_KEY=...
    python app.py
"""

import os
import time

import matplotlib
matplotlib.use("Agg")  # headless backend, must be set before pyplot is ever imported

from flask import Flask, request, jsonify, render_template

from test_generation_agent import TestGenerationAgent

app = Flask(__name__)

agent = TestGenerationAgent(
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    model_name=os.environ.get("GEMMA_MODEL_NAME", "gemma-4-26b-a4b-it"),
    persist_directory=os.environ.get("CHROMA_DIR", "./chroma_store"),
    collection_name="test_examples",
)

# Tracks the most recently generated example so the "Submit answer" button
# knows which stored document to attach feedback to. Single-session demo state.
last_generated = {"id": None, "question": None, "answer": None, "concept": None}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True) or {}
    concept = (data.get("concept") or "").strip()
    questions = [q.strip() for q in data.get("questions", []) if q and q.strip()]
    answers = [a.strip() for a in data.get("answers", []) if a and a.strip()]

    if not concept:
        return jsonify({"error": "Concept is required."}), 400
    if len(questions) < 2 or len(answers) < 2 or len(questions) != len(answers):
        return jsonify({"error": "Provide 2-3 matched example questions and answers."}), 400

    few_shot_examples = [{"question": q, "answer": a} for q, a in zip(questions, answers)]

    try:
        result = agent.generate_example(concept, few_shot_examples)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    last_generated["id"] = result["id"]
    last_generated["question"] = result["question"]
    last_generated["answer"] = result["answer"]
    last_generated["concept"] = concept

    return jsonify({
        "id": result["id"],
        "question": result["question"],
        "attempts": result["attempts"],
        "accepted_on_similarity_check": result["accepted_on_similarity_check"],
    })


@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(force=True) or {}
    feedback = (data.get("feedback") or "").strip()
    user_answer = (data.get("user_answer") or "").strip()

    if not last_generated["id"]:
        return jsonify({"error": "Generate an example before submitting feedback."}), 400
    if not feedback:
        return jsonify({"error": "Select a feedback option."}), 400

    try:
        agent.record_feedback(last_generated["id"], feedback=feedback, user_answer=user_answer)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok"})


@app.route("/visual", methods=["POST"])
def visual():
    static_dir = os.path.join(app.root_path, "static")
    os.makedirs(static_dir, exist_ok=True)
    filename = f"plot_{int(time.time())}.png"
    save_path = os.path.join(static_dir, filename)

    try:
        agent.plot_summary(batch_size=10, save_path=save_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"url": f"/static/{filename}"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
