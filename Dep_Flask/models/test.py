from flask import Flask, request, jsonify

app = Flask (__name__)

@app.route("/test", methods=["GET"])
def test():
    if request.method == "GET":
        return jsonify({"response": "Get Request Clearly"})


if __name__ == '__main__':
    app.run(debug=True, port=8080)