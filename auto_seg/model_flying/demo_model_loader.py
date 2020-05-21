import os
import shutil

from tools import auto_run

EXP_FOLDER = './records'

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
        if self.ckpt_path:
            # TODO 检查文件夹中有没有model和超参文件
            pass

        elif os.path.isdir(EXP_FOLDER):
            folders = os.listdir(EXP_FOLDER)
            if folders:
                self.ckpt_path = os.path.join(EXP_FOLDER, folders[0])

        if self.ckpt_path:
            self.status = 'trained'


    def inference(self, image):
        try:
            # label_name, max_prob = self.infer_job.run(image)
            # TODO 调用推理接口

            return 'class: %s, prob: %.5f' % (label_name, max_prob)

        except Exception as e:
            print(e)
            return 'image data or model error'


    def train(self, dataset_path):
        ckpt_path = auto_run.trainer(dataset_path, EXP_FOLDER)
        if not ckpt_path:
            return False

        if self.ckpt_path and self.inferer:
            del self.inferer
            shutil.rmtree(self.ckpt_path)

        self.ckpt_path = ckpt_path
        self.__load_models()

        print(self.ckpt_path)

        return True


    def test(self, dataset_path):
        if not self.ckpt_path:
            return

        # TODO 调用测试
        results, _ = auto_run.test(dataset_path, self.ckpt_path)

        test_metrics_results = results[0][0]
        test_metrics_mean = {key:test_metrics_results[key][0] for key in test_metrics_results.keys()}

        print(test_metrics_mean, flush=True)

        return test_metrics_mean


    def get_status(self):
        return self.status


    def set_status(self, status):
        self.status = status

