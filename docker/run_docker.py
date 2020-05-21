import time
import os
import ntplib

# 权限粒度控制，time：只控制时间权限；host：只控制运行服务器权限；time&host：控制时间和运行服务器权限
AUTH_CONTROL_TYPES = {"time", "host", "time&host"}

# 权限控制开关,设置为True为打开，False为关闭权限控制
AUTH_CONTROL = False
AUTH_CONTROL_TYPE = "time&host"
# 服务器物理网卡MAC地址
HOST_MAC_ADDR = "ac:1f:6b:95:c6:25"
# 授权开始时间
AUTH_START_TIME='2020-3-16 14:20'
# 总共授权时间天数
AUTH_TIME = 100

DOCKER_IMAGE = "autocv/segment:v0.1"
CONTAINER_NAME = "autocv_segment_v_0_1_contrainer"
DATA_DES_DIR = "/home/root/dataset"
RECORD_DST_DIR = "/home/root/records"


def __time_auth_control():
    if not AUTH_CONTROL:
        print('authorization control is off')
        return True

    auth_start_time_secs = time.mktime(time.strptime(AUTH_START_TIME, "%Y-%m-%d %H:%M"))
    auth_time = AUTH_TIME * 3600 * 24

    try:
        # get web time
        c = ntplib.NTPClient()
        # use China timezone
        resp = c.request('ntp.ntsc.ac.cn', version = 3)

        if (auth_start_time_secs + auth_time) > resp.tx_time:
            return True

        print('authorization timeout !!!')
        return False
    except Exception as e:
        print(e)
        print('get web time error, can not use this application !!!')
        return False


def __host_auth_control():
    net_path = "/sys/class/net"
    re =  os.popen("ls {}".format(net_path)).readlines()
    if len(re) == 0:
        return False
    re = [r.strip() for r in re if r.strip not in {"lo", "docker0"}]
    mac_addrs = set()
    for r in re:
        cmd =  "cat {}/{}/address".format(net_path, r)
        mac = os.popen(cmd).readlines()
        if len(mac) == 0:
            return False
        mac_addrs.add(mac[0].strip())

    if HOST_MAC_ADDR not in mac_addrs:
        return False

    return True


def auth():
    if AUTH_CONTROL:
        if AUTH_CONTROL_TYPE == "host" and not __host_auth_control():
            print("当前运行的主机不是程序授权的主机，请在指定服务器上运行或联系管理员")
            return False
        elif AUTH_CONTROL_TYPE == "time" and not __time_auth_control():
            print("授权时间已过期或主机网络不可用，请联系管理员")
            return False
        elif AUTH_CONTROL_TYPE == "time&host":
            if not __time_auth_control():
                print("授权时间已过期或主机网络不可用，请联系管理员")
                return False
            elif not __host_auth_control():
                print("当前运行的主机不是程序授权的主机，请在指定服务器上运行或联系管理员")
                return False

    return True


def kill_delete_container():
    cmd = "docker stop {} && docker rm {}".format(CONTAINER_NAME, CONTAINER_NAME)
    try:
        os.system(cmd)
    except Exception as e:
        print(e)


def start_train(args):
    if not auth():
        exit(-1)

    kill_delete_container()

    # dev_mount = "-v {}:{} -v {}:{} -v {}:{}".format(args.dataset_path,
    dev_mount = "-v {}:{} -v {}:{}".format(args.dataset_path,
                                                   DATA_DES_DIR,
                                                   os.path.join(os.path.abspath("."), "records"),
                                                   RECORD_DST_DIR,)
                                                   # os.path.join(os.path.abspath("."), "logs"),
                                                   # LOGS_DST_DIR)
    start_cmd = "docker run --name {} --shm-size=32G --gpus all {} {} " \
                "bash -c \"python -c \\\"from autocv_classification_pytorch import autocv;" \
                "autocv.train('{}',available_gpus='{}',other_hp_search=False)\\\"\"".format(
                                                                            CONTAINER_NAME,
                                                                             dev_mount,
                                                                             DOCKER_IMAGE,
                                                                             DATA_DES_DIR,
                                                                             args.visible_devices_list)
    os.system(start_cmd)


def start_test(args):
    if not auth():
        exit(-1)

    kill_delete_container()
    dev_mount="-v {}:{} -v {}:{}".format(args.dataset_path,
                                           DATA_DES_DIR,
                                           args.model_path,
                                           RECORD_DST_DIR)
    start_cmd = "docker run --name {} --shm-size=32G --gpus all {} {} " \
                "bash -c \"python -c \\\"from autocv_classification_pytorch import autocv;" \
                "autocv.test('{}','{}',available_gpus='{}')\\\"\"".format(CONTAINER_NAME,
                                                                         dev_mount,
                                                                         DOCKER_IMAGE,
                                                                         DATA_DES_DIR,
                                                                         RECORD_DST_DIR,
                                                                         args.visible_devices_list)
    os.system(start_cmd)


def start_server(args):
    if not auth():
        exit(-1)

    kill_delete_container()

    # dev_mount = "-v {}:{} -v {}:{} -v {}:{}".format(args.model_path,
    dev_mount = "-v {}:{} -v {}:{}".format(args.model_path,
                                            RECORD_DST_DIR,
                                            # os.path.join(os.path.abspath("."), "logs"),
                                            # LOGS_DST_DIR,
                                            os.path.join(os.path.abspath("."), "data"),
                                            DATA_DES_DIR)
    start_cmd = "docker run --name {} -p {}:{} --shm-size=32G --gpus all {} {} " \
                "python tools/start_server.py --port {}".format(CONTAINER_NAME,
                                                                        args.port,
                                                                        args.port,
                                                                        dev_mount,
                                                                        DOCKER_IMAGE,
                                                                        args.port)
    # print(start_cmd)
    os.system(start_cmd)
