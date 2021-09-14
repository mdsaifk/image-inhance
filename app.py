from flask import Flask, render_template, Response, request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import model_func
from wsgiref.simple_server import make_server
from wsgiref import simple_server
import base64
import io, json
from models.experimental import *
from utils.datasets import *

from PIL import Image
import os
import sys
import cv2
#from app_helper import *

__author__ = 'Meenu'
__source__ = ''

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # create a secure filename
        filename = secure_filename(f.filename)
        print("**************************")
        print(filename)
        # save file to /static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(filepath)
        f.save(filepath)
        #get_image(filepath, filename)
        wt_path='runs/exp5_yolov5s_results/weights/best.pt'
        model_func.detect(wt_path)
        return render_template("uploaded.html", display_detection=filename, fname=filename)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

#if __name__ == '__main__':
    #app.run(port=7000, debug=True)

if __name__ == "__main__":
    #cliApp = Api()
    port = int(os.getenv("PORT"))
    host = '0.0.0.0'
    #port = 7000
    httpd = simple_server.make_server(host, port=port, app=app)
    #print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
    