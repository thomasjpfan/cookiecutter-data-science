"""File Locations"""
import os


class RawFiles:

    def __init__(self, data_dir):
        self.raw_dir = os.path.join(data_dir, "raw")


class ProcessFiles:

    def __init__(self, data_dir):
        self.proc_dir = os.path.join(data_dir, "processed")
