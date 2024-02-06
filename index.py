from flask import flask,render_template

app=flask(_name_)
@app.route("/")
def home():
    return render_template('index.html')

if _name_="_main_":
    app.run(debug=false,host='0.0.0.0')
