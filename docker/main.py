import argparse
from run_docker import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="欢迎使用AutoCV程序")
    sub_parser = parser.add_subparsers(help="本程序用于加载并运行AutoCV容器内部功能, 包括训练、批量测试、推理以及模型服务化")

    train_parser = sub_parser.add_parser("train", help = "启动模型训练")
    train_parser.add_argument("--subcommand", default = "train")
    train_parser.add_argument("--dataset_path", type = str, help = "数据集绝对路径", required = True)
    train_parser.add_argument("--exp_code", type = str, default = "fix", help = "测试ID")
    train_parser.add_argument("--visible_devices_list", type = str,
                              default = "0,1,2,3", help = "逗号隔开的可见GPU ID列表，默认0,1,2,3")

    test_parser = sub_parser.add_parser("test", help = "启动模型批量测试")
    test_parser.add_argument("--subcommand", default = "test")
    test_parser.add_argument("--model_path", type = str, help = "已训练模型绝对路径", required = True)
    test_parser.add_argument("--dataset_path", type = str, help ="数据集绝对路径", required = True)
    test_parser.add_argument("--visible_devices_list", type = str,
                             default = "0,1,2,3", help = "逗号隔开的可见GPU ID列表，默认0,1,2,3")

    start_server_parser = sub_parser.add_parser("start_server", help="启动模型服务，为通过网络进行预测提供服务")
    start_server_parser.add_argument("--subcommand", default="start_server")
    start_server_parser.add_argument("--model_path", type=str, help = "本地存储训练模型的绝对路径", required = True)
    start_server_parser.add_argument("--port", type=int, default = 5000, help = "服务端口，默认5000")
    start_server_parser.add_argument("--visible_devices_list", type = str,
                                     default='0,1,2,3', help = "逗号隔开的可见GPU ID列表，默认0,1,2,3")

    args,_ = parser.parse_known_args()
    if "subcommand" in args:
        if args.subcommand == "train":
            start_train(args)
        elif args.subcommand == "test":
            start_test(args)
        # elif args.subcommand == "inference":
        #     start_inference(args)
        elif args.subcommand == "start_server":
            start_server(args)
