from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Result Predictor</title>
</head>
<body>
    <h1>Student Result Predictor</h1>
    <p>Model Version: 1.0</p>

    <form method="POST">
        Study Hours:
        <input type="number" name="hours"
               min="0" max="24" step="any" required>
        <br><br>

        Attendance (%):
        <input type="number" name="attendance"
               min="0" max="100" step="any" required>
        <br><br>

        Internal Marks:
        <input type="number" name="internal"
               min="0" max="100" step="any" required>
        <br><br>

        <button type="submit">Predict Result</button>
    </form>

    {% if result %}
        <h2>Predicted Result: {{ result }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        hours = float(request.form["hours"])
        attendance = float(request.form["attendance"])
        internal = float(request.form["internal"])

        inputs = pd.DataFrame(
            [[hours, attendance, internal]],
            columns=["study_hours", "attendance", "internal"]
        )

        prediction = model.predict(inputs)[0]
        result = "Pass" if prediction == 1 else "Fail"

    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)