import os


def create_multiple_dirs(paths: list):
    for path in paths:
        custom_mkdir(path)


def custom_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
