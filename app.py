from flask import Flask, jsonify, render_template, request

from recommender_service import QueryError, recommend

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    profile = request.get_json(silent=True)
    if not isinstance(profile, dict):
        return jsonify({"error": "Request body must be a JSON search profile."}), 400
    try:
        return jsonify(recommend(profile))
    except QueryError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": f"Required pipeline output missing: {exc}"}), 500
    except Exception as exc:  # keep the browser on JSON even for unexpected errors
        app.logger.exception("Recommendation failed")
        return jsonify({"error": f"Recommendation failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
