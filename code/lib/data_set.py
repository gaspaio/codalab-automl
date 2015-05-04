""" DataSet object definition. """

import os


class DataSet:

    """ Load & save datasets. """

    def __init__(self, basename, config):
        """ Constructor. """
        self.basename = basename,
        self.input_dir = os.path.join(config["data_dir"], basename),
        self.info = self.getInfoFromFile(
            os.path.join(config["data_dir"], basename + "_public.info"))



    def getInfoFromFile(self, filename):
        """ Load all information pairs from the public.info file. """
        info = {}
        with open(filename, "r") as info_file:
            lines = info_file.readlines()
            features_list = list(map(
                lambda x: tuple(x.strip("\'").split(" = ")), lines))

            for (key, value) in features_list:
                info[key] = value.rstrip().strip("'").strip(' ')

                # if we have a number, we want it to be an integer
                if info[key].isdigit():
                    info[key] = int(info[key])

        return info
