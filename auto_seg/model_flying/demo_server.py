from model_flying.demo_model_loader import Model_Loader
from utils.utils import check_data_format
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import zipfile
import os


train_data_path = ''
test_data_path = ''
inference_video_path=''
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


@app.route('/segment/upload/<string:t>', methods=['POST'])
def upload(t, **kwargs):
    if request.method == 'POST':
        file = request.files['file_' + t]
        if file:
            if t != 'inference_video':
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
                    check_data_format(data_path, True if t == 'train' else False)
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
            else:
                filename = secure_filename(file.filename)
                data_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(data_path)

                global inference_video_path
                inference_video_path = data_path

                print('inference_video_path: {}'.format(inference_video_path), flush=True)

            return {'return': 'success'}

    return {'return': 'error'}


@app.route('/segment/train', methods=['POST'])
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


@app.route('/segment/test', methods=['POST'])
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


@app.route('/segment/get_status', methods=['GET'])
def get_status():
    global test_results
    status = model.get_status()
    if status == 'tested':
        model.set_status('trained')
        return {'return': 'tested', 'records': str(test_results)}

    return {'return': status}


@app.route('/segment/inference', methods=['POST'])
def inference():
    f = request.files['file_inference']
    data = f.read()
    return {'return': 'success', 'predict' : model.inference(data)}


@app.route('/segment/inference_video', methods=['POST'])
def inference_video():
    global inference_video_path
    if not inference_video_path:
        return {'return': 'no data'}

    model.set_status('testing')

    sub_folder = 'static/video'
    full_folder = os.path.join(os.path.split(__file__)[0], sub_folder)
    if not os.path.isdir(full_folder):
        os.makedirs(full_folder)
    output_video_name = 'inference_video.avi'
    tmp_file_path = os.path.join(full_folder, output_video_name)

    ret = model.inference_video(inference_video_path, tmp_file_path)
    model.set_status('tested')
    if ret:
        return {'return': 'success', 'records': os.path.join(sub_folder, output_video_name)}
    else:
        return {'return': 'failed', 'records': 'write video file error'}



def start_server(port, visible_devices_list):
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    app.config.from_mapping(
        UPLOAD_FOLDER = UPLOAD_FOLDER,
    )

    print("loading models...", flush=True)
    global model
    model = Model_Loader(visible_devices_list)

    print("server run...", flush=True)
    app.run(host='0.0.0.0', port = port, debug = True,  use_reloader = False)
    # server = WSGIServer(('0.0.0.0', port), app)#.serve_forever()
    # server.start()

