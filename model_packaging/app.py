from flask import Flask, request, jsonify, send_from_directory , render_template
import model_infer  # Import your model inference script
from flask_cors import CORS
import json

  
app = Flask(__name__)
CORS(app)  

  
@app.route('/')  
def index():  
    return render_template('index.html') 
  
@app.route('/infer', methods=['POST'])  
def infer():
    try: 
        predicted_class = model_infer.run_inference()
        print(predicted_class)
        print(predicted_class.tolist())
        json_data = json.dumps(predicted_class.tolist())  
        return jsonify({'result': json_data})  
    except Exception as e:  
        return jsonify({'error': str(e)}), 500  
  
if __name__ == '__main__':  
    app.run(debug=True)