# coding=utf-8
import argparse
import requests
import json
import time


def train(dataset_path, ip = '0.0.0.0', port = 5001):
    # 先上传数据，如果数据较大上传时间可能会很长
    upload_url = 'http://{}:{}/segment/upload/train'.format(ip, port)
    with open(dataset_path, 'rb') as f:
        r = requests.post(upload_url, files={'file_train': f})
        print(r.text)
        format_ret = json.loads(r.text)
        # 如果接口有返回，但是返回内容不是success的话，说明上传失败，失败的字段有dataset invalid、zip file error和error
        # 其中dataset invalid和zip file error表示数据格式存在错误，error表示其他错误
        if format_ret['return'] != 'success':
            return
    print('upload dataset success')

    # 发起训练
    train_url = 'http://{}:{}/segment/train'.format(ip, port)
    # 一般训练时间很长，这个接口返回可能需要很久，数据量大的话可能需要几天，此处建议设置timeout,
    # 超时后，通过轮询请求目前系统的状态
    try:
        r = requests.post(train_url, timeout = 30)
        format_ret = json.loads(r.text)
        if format_ret['return'] == 'no data':
            print('no data')
            return
    except requests.exceptions.Timeout:
        print('request timeout')
    # 超时轮询检查是否训练完成，后台返回的状态类型有training、trained、non-trained、testing和tested，
    # training: 表示正在训练中，训练中不能发送批量测试、推理和再次训练
    # trained: 表示训练完成，训练完成后才可以发送批量测试、推理
    # non-trained: 表示没有训练，后台正在闲置，没有训练不能发送批量测试、推理，只能发送训练请求
    # testing: 表示正在批量测试中，批量测试中不能发送再次批量测试、推理和再次训练
    # tested: 表示正在批量测试完成，这个状态是瞬间状态，只在调用批量测试接口后返回
    status_url = 'http://{}:{}/segment/get_status'.format(ip, port)
    while True:
        r = requests.get(status_url)
        print(r)
        format_ret = json.loads(r.text)
        if format_ret['return'] == 'trained':
            print('训练完成')
            break
        elif format_ret['return'] == 'non-trained':
            print('训练失败')
            break
        elif format_ret['return'] == 'training':
            print('训练中...')
            # 由于训练时间较长，建议不要频繁请求状态
            time.sleep(60)


def test(dataset_path, ip = '0.0.0.0', port = 5001):
    # 先上传数据，如果数据较大上传时间可能会很长
    upload_url = 'http://{}:{}/segment/upload/test'.format(ip, port)
    with open(dataset_path, 'rb') as f:
        r = requests.post(upload_url, files={'file_test': f})
        print(r.text)
        format_ret = json.loads(r.text)
        # 如果接口有返回，但是返回内容不是success的话，说明上传失败，失败的字段有dataset invalid、zip file error和error
        # 其中dataset invalid和zip file error表示数据格式存在错误，error表示其他错误
        if format_ret['return'] != 'success':
            return
    print('upload dataset success')

    # 发起测试
    test_url = 'http://{}:{}/segment/test'.format(ip, port)
    # 一般测试时间略长
    r = requests.post(test_url)
    format_ret = json.loads(r.text)
    if format_ret['return'] == 'no data':
        print('no data')
        return
    # records字段包含测试结果
    records = format_ret['records']
    print(records)


def segment(img, ip = '0.0.0.0', port = 5001):
    req_url = 'http://{}:{}/segment/inference'.format(ip, port)

    # 发起推理请求
    with open(img, 'rb') as f:
        r = requests.post(req_url, files = {'file_inference' : f})

    # 下载推理标签图片文件
    results = json.loads(r.text)
    download_url = 'http://{}:{}/{}'.format(ip, port, results['predict'])
    r = requests.get(download_url)
    save_file_name = img.split('/')[-1].split('.')[0] + '_predict.png'
    with open(save_file_name, 'wb') as f:
        f.write(r.content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="远程访问模型进行图片预测工具")
    sub_parser = parser.add_subparsers(help='选择不同功能')

    train_parser = sub_parser.add_parser('train', help='训练')
    train_parser.add_argument('--subcommand', default='train')
    train_parser.add_argument('--file', help="图片数据集zip格式压缩包路径，必须提供", required=True)
    train_parser.add_argument('--ip', help="服务IP，默认 0.0.0.0", type = str, default = '0.0.0.0', required = False)
    train_parser.add_argument('--port', help="服务端口号，默认 5001", type = int, default = 5001, required = False)

    test_parser = sub_parser.add_parser('test', help='测试')
    test_parser.add_argument('--subcommand', default='test')
    test_parser.add_argument('--file', help="图片数据集zip格式压缩包路径，必须提供", required=True)
    test_parser.add_argument('--ip', help="服务IP，默认 0.0.0.0", type = str, default = '0.0.0.0', required = False)
    test_parser.add_argument('--port', help="服务端口号，默认 5001", type = int, default = 5001, required = False)

    infer_parser = sub_parser.add_parser('infer', help='推理')
    infer_parser.add_argument('--subcommand', default='infer')
    infer_parser.add_argument('--image', help="图片的本地地址，必须提供", required=True)
    infer_parser.add_argument('--ip', help="服务IP，默认 0.0.0.0", type = str, default = '0.0.0.0', required = False)
    infer_parser.add_argument('--port', help="服务端口号，默认 5001", type = int, default = 5001, required = False)

    args, _ = parser.parse_known_args()

    if args.subcommand == 'train':
        train(args.file, args.ip, args.port)
    elif args.subcommand == 'test':
        test(args.file, args.ip, args.port)
    elif args.subcommand == 'infer':
        predict = segment(args.image, args.ip, args.port)
        print(predict)

