from model_flying.demo_model_loader import Model_Loader
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import zipfile
import os


train_data_path = ''
test_data_path = ''
UPLOAD_FOLDER = '/home/root/dataset'
test_results = {}

app = Flask(__name__)
model = None


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
        return True

    print('This is not zip file')
    return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify/upload/<string:t>', methods=['POST'])
def upload(t, **kwargs):
    if request.method == 'POST':
        file = request.files['file_' + t]
        if file:
            filename = secure_filename(file.filename)
            data_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(data_path)

            # unzip dataset
            if not unzip_file(data_path, app.config['UPLOAD_FOLDER']):
                print('unzip error')
                return {'return': 'zip file error'}

            # dataset directory
            data_path = '.'.join(data_path.split('.')[0:-1])
            # check data validity
            try:
                # onefile_dataset_validity_check(data_path, 'train.txt')
                pass
            except Exception as e:
                print(e)
                return {'return': 'dataset invalid'}

            if t == 'train':
                global train_data_path
                train_data_path = data_path
                print(train_data_path)
            else:
                global test_data_path
                test_data_path = data_path
                print(test_data_path)

            return {'return': 'success'}

    return {'return': 'error'}


@app.route('/classify/train', methods=['POST'])
def train():
    global train_data_path
    if not train_data_path:
        return {'return': 'no data'}

    model.set_status('training')

    if model.train(train_data_path):
        model.set_status('trained')
        return {'return': 'success'}

    model.set_status('non-trained')
    return {'return': 'failed'}


@app.route('/classify/test', methods=['POST'])
def test():
    global test_data_path
    global test_results
    if not test_data_path:
        return {'return': 'no data'}

    model.set_status('testing')
    test_results = model.test(test_data_path)
    model.set_status('tested')
    if test_results:
        return {'return': 'success', 'records': str(test_results)}

    return {'return': 'failed', 'records': ''}


@app.route('/classify/get_status', methods=['GET'])
def get_status():
    global test_results
    status = model.get_status()
    if status == 'tested':
        model.set_status('trained')
        return {'return': 'tested', 'records': str(test_results)}

    return {'return': status}


@app.route('/classify/inference', methods=['POST'])
def inference():
    f = request.files['file_inference']
    data = f.read()
    return {'return': 'success', 'predict' : model.inference(data)}


def start_server(port):
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    app.config.from_mapping(
        UPLOAD_FOLDER = UPLOAD_FOLDER,
    )

    print("loading models...", flush=True)
    global model
    model = Model_Loader()

    print("server run...", flush=True)
    app.run(host='0.0.0.0', port = port, debug = True,  use_reloader = False)
    # server = WSGIServer(('0.0.0.0', port), app)#.serve_forever()
    # server.start()

