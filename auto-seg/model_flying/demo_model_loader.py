import os
import shutil

from autocv_classification_pytorch.policy.jobs.swa_cyclic_mpncov_jobs import SwaMpncovInferenceJob
from autocv_classification_pytorch.model_tunner.settings import Settings
from autocv_classification_pytorch import autocv

EXP_FOLDER = './records'

class Model_Loader(object):
    def __init__(self, visible_devices):
        """
        model loader for web demo
        """
        self.ckpt_path = None
        self.visible_devices = visible_devices
        self.config = dict()
        self.infer_job = None

        self.status = 'non-trained'

        self.__load_models()


    def __load_models(self):
        if self.ckpt_path:
            # train sub-folder name is number, e.g. 0/1
            fs = [f for f in os.listdir(self.ckpt_path) if f.isdigit()]
            if len(fs) == 0:
                print("There is no model folder")
                return
        elif os.path.isdir(EXP_FOLDER):
            folders = os.listdir(EXP_FOLDER)
            if folders:
                self.ckpt_path = os.path.join(EXP_FOLDER, folders[0], 'saved_models/')

        if self.ckpt_path:
            self.status = 'trained'
            settings = Settings(available_gpus = self.visible_devices)
            self.infer_job = SwaMpncovInferenceJob(settings, self.ckpt_path)


    def inference(self, image):
        try:
            label_name, max_prob = self.infer_job.run(image)

            return 'class: %s, prob: %.5f' % (label_name, max_prob)

        except Exception as e:
            print(e)
            return 'image data or model error'


    def train(self, dataset_path):
        ckpt_path = autocv.train(dataset_path,
                                 other_hp_search = False,
                                 available_gpus = self.visible_devices,
                                 records_root_dir = EXP_FOLDER)
        if not ckpt_path:
            return False

        if self.ckpt_path and self.infer_job:
            del self.infer_job
            shutil.rmtree(os.path.split(self.ckpt_path)[0])

        self.ckpt_path = ckpt_path
        self.__load_models()

        print(self.ckpt_path)

        return True


    def test(self, dataset_path):
        if not self.ckpt_path:
            return

        results, _ = autocv.test(dataset_path,
                              self.ckpt_path,
                              available_gpus = self.visible_devices)

        test_metrics_results = results[0][0]
        test_metrics_mean = {key:test_metrics_results[key][0] for key in test_metrics_results.keys()}

        print(test_metrics_mean, flush=True)

        return test_metrics_mean


    def get_status(self):
        return self.status


    def set_status(self, status):
        self.status = status

