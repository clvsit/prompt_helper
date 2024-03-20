import os
import subprocess

from rich import print
from rich.panel import Panel
from loguru import logger


def get_root_dir() -> str:
    cur_path = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件的目录
    proj_path = cur_path[: cur_path.find("core")]  # 获取根目录
    return proj_path


class Consoler:
    @staticmethod
    def print_in_panel(text: str, title: str = "", subtitle: str = ""):
        if title and subtitle:
            print(Panel(text, title=title, subtitle=subtitle))
        elif title:
            print(Panel(text, title=title))
        else:
            print(Panel(text))


class SystemCommander:
    @staticmethod
    def execute(command_str: str) -> str:
        """
        执行系统命令
        :param command_str: str 命令字符串
        :return 命令在控制台的输出文本
        """
        result = subprocess.Popen(
            command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        raw_info = str(result.stdout.read(), "UTF-8")
        return raw_info


def get_logger(log_dir=None, format_=None):
    if log_dir is None:
        log_dir = "./logs"
    if format_ is None:
        format_ = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <yellow>{extra["
            "request_id]}</yellow> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{"
            "message}</level> "
        )
    log_file_path = os.path.join(log_dir, "info")
    err_file_path = os.path.join(log_dir, "error")

    extra = {"request_id": "default"}
    logger.configure(extra=extra)
    if format_ is None:
        format_ = "{time} {level} {message}"

    # formatter = Formatter()
    # format_ = formatter.get_format()
    # rotator = Rotator(size=1e+8, at=datetime.time(0, 0, 0))
    # logger.remove(0)
    # logger.add(sys.stderr, format=format_, diagnose=True)
    # logger.add(log_file_path, format=formatter.get_format(), rotation=rotator.should_rotate, encoding='utf-8',
    #            enqueue=True, diagnose=True)
    # logger.add(err_file_path, format=formatter.get_format(), rotation=rotator.should_rotate, encoding='utf-8',
    #            level='ERROR',
    #            enqueue=True, diagnose=True)
    #
    # logger.opt(exception=True)
    # logger.add(sys.stdout, serialize=True)
    return logger


logger_ = get_logger()
