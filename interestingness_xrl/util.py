__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import os
import numpy as np
from PIL.Image import fromarray
from logging import Logger
from gym.wrappers.monitor import Monitor
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from collections import OrderedDict


def clean_console():
    """
    Cleans the system's console using the 'cls' or 'clear' command.
    :return:
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def log_and_print(msg, log):
    """
    Prints a message to the log (into level) and also to the console.
    :param str msg: the messaged to be logged / displayed.
    :param Logger log: the log in which to save the message.
    :return:
    """
    log.info(msg)
    print(msg)


def print_line(msg, file=None, write_console=True, newline=True):
    """
    Prints a message to a file as a line and optionally to the console.
    :param str msg: the message to be saved / displayed.
    :param stream file: the file on which to save the message line.
    :param bool write_console: whether to print the message to the screen.
    :param bool newline: whether to write a new line character at the end of printing the message.
    :return:
    """
    if file is not None:
        if newline:
            file.write(msg + '\n')
        else:
            file.write(msg)
        if not write_console:
            return
    if newline:
        print(msg)
    else:
        print(msg, end='', flush=True)


def record_video(buffer, file_path, fps):
    """
    Records the given image buffer to a video file.
    :param array_like buffer: the list containing the sequential image frames to be recorded.
    :param str file_path: the path to the video file.
    :param int fps: the video frames-per-second.
    :return:
    """

    # creates video encoder and adds each frame in the buffer
    video_recorder = ImageEncoder(file_path, buffer[0].shape, fps, fps)
    for frame in buffer:
        video_recorder.capture_frame(frame)
    video_recorder.close()


def save_image(env, file_path):
    """
    Saves a screenshot of the given Gym environment monitor.
    :param Monitor env: the Gym environment monitor from which to extract the image.
    :param str file_path: the path to the image file.
    :return:
    """
    image_rotated = np.fliplr(np.rot90(env.env.env.game_state.getScreenRGB(), 3))
    fromarray(image_rotated).save(file_path)


def dict_to_list(dct):
    """
    Converts the given dictionary into a list of key-value pairs.
    :param dict dct: the dictionary to be converted.
    :return list: a list of key-value tuples.
    """
    return [(k, v) for k, v in dct.items()]


def list_to_dict(lst):
    """
    Converts the given list of key-value tuples into an ordered dictionary.
    :param list lst: a list of key-value tuples to be converted.
    :return OrderedDict: an ordered dictionary containing the elements in the list.
    """
    dct = OrderedDict()
    for k, v in lst:
        dct[k] = v
    return dct


def clean_directory(dir_path):
    """
    Deletes all files in the given directory.
    :param str dir_path: the path to the directory we want to clean.
    :return:
    """
    for the_file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
