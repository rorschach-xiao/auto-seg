import os
import shutil
import cv2
import logging

from tools import auto_run

EXP_FOLDER = './records'

def has_model_params_file_under_folder(ckpt_path):
    final_state_file = os.path.join(ckpt_path, 'final_state.pth')
    params_file = os.path.join(ckpt_path, 'param.json')

    if os.path.isfile(final_state_file) and os.path.isfile(params_file):
        return True
    return False


class Model_Loader(object):
    def __init__(self):
        """
        model loader for web demo
        """
        self.ckpt_path = None
        self.inferer = None

        self.status = 'non-trained'

        self.__load_models()


    def __load_models(self):
        if not self.ckpt_path and os.path.isdir(EXP_FOLDER):
            folders = os.listdir(EXP_FOLDER)
            if folders:
                self.ckpt_path = os.path.join(EXP_FOLDER, folders[0])

        if self.ckpt_path and has_model_params_file_under_folder(self.ckpt_path):
            self.inferer = auto_run.InferenceJob(self.ckpt_path)
            self.status = 'trained'

            return True

        self.ckpt_path = None
        return False


    def inference(self, image):
        try:
            tmp_file_path = 'static/img/inference_results.jpg'
            output_img = self.infer_job.run(image)
            cv2.imwrite(output_img, tmp_file_path)

            return tmp_file_path
        except Exception as e:
            print(e, flush = True)
            logging.exception(e)
            return 'image data or model error'


    def train(self, dataset_path):
        try:
            ckpt_path = auto_run.train(dataset_path, EXP_FOLDER)
        except Exception as e:
            print(e, flush = True)
            logging.exception(e)
            return False

        if not ckpt_path:
            return False

        if self.ckpt_path and self.inferer:
            del self.inferer
            shutil.rmtree(self.ckpt_path)

        self.ckpt_path = ckpt_path
        if self.__load_models():
            print(self.ckpt_path, flush=True)
            return True

        return False


    def test(self, dataset_path):
        if not self.ckpt_path:
            return

        results = auto_run.test(dataset_path, self.ckpt_path)

        print(results, flush=True)
        return results


    def get_status(self):
        return self.status


    def set_status(self, status):
        self.status = status

