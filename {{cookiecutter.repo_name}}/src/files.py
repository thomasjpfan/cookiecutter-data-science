"""File Locations"""
import os


class RawFiles:

    def __init__(self, data_dir):
        raw_dir = os.path.join(data_dir, "raw")
        pass


class ProcessFiles:

    def __init__(self, data_dir):
        proc_dir = os.path.join(data_dir, "processed")
        pass
