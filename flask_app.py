import os
import Extract_similarity3 as es
from flask import Flask,request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def get_info():
    #data = request.json
    data = request.get_json(force=True)
    cv_text = data['cv_text']
    desc_text = data['desc_text']
    json_out= es.get_info(cv_text,desc_text)
    return json_out


if __name__ == "__main__":
    #app.run(debug=True, port=5500)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    
