#!/usr/bin/env python
""" AutoML code.

TODO:
- Add time budget


"""

import os
from sys import path, argv
import time
import datetime as dt
from config import CONFIG

path.append(os.path.join(CONFIG['root_dir'], "vendor"))
path.append(os.path.join(CONFIG['root_dir'], "lib"))

from data_manager import DataManager
import data_io
from auto_ml import AutoML

the_date = dt.datetime.now().strftime("%y-%m-%d-%H-%M")
res_dir = os.path.join(CONFIG['root_dir'], "res")


def predict(datanames, input_dir):
    """ Main function. """
    overall_time_budget = 0
    res_dir = os.path.join(CONFIG['root_dir'], "res")

    for basename in datanames:
        print "\n*** Processing dataset %s" % basename.upper()
        start = time.time()

        D = DataManager(basename,
                        input_dir,
                        replace_missing=False,
                        filter_features=False,
                        verbose=False)

        # Set overall time budget with this dataset's allocated time
        time_budget = int(0.8 * D.info['time_budget'])

        overall_time_budget = overall_time_budget + time_budget
        read_time = time.time() - start
        ts = time.time()

        aml = AutoML(D, CONFIG)
        aml.run_predict(time_budget)

        run_time = time.time() - ts
        end = time.time()

        print "* Time:: budget=%5.2f, load=%5.2f, run=%5.2f, remaining=%5.2f" \
            % (time_budget, read_time, run_time, time_budget - (end - start))

        for i, res in enumerate(aml._Y):
            filename = basename + "_valid_" + str(i).zfill(3) + ".predict"
            data_io.write(
                os.path.join(res_dir, filename), aml._Y[i]['Y_valid'])
            filename = basename + "_test_" + str(i).zfill(3) + ".predict"
            data_io.write(
                os.path.join(res_dir, filename), aml._Y[i]['Y_test'])

    return True

if __name__ == "__main__":
    # Input / Output
    if len(argv) == 1:
        input_dir = os.path.join(CONFIG['root_dir'], '..', 'data')
        output_dir = CONFIG['default_output_dir']
    else:
        input_dir = argv[1]
        output_dir = os.path.abspath(argv[2])

    data_io.mvdir(output_dir, output_dir + '_' + the_date)
    data_io.mkdir(output_dir)

    datanames = data_io.inventory_data(input_dir)
    # datanames = ["madeline"]

    # Result submission : copy files from /res to output dir
    OK = data_io.copy_results(datanames, res_dir, output_dir, False)
    if OK:
        print "Result files copied output dir. Not running train code."
        execution_success = True
    else:
        execution_success = predict(datanames, input_dir)

    if execution_success and CONFIG["zipme"]:
        submission_filename = "../automl_{}_{}.zip".format(CONFIG['codename'], the_date)
        data_io.zipdir(submission_filename, ".")

    if CONFIG['running_on_codalab']:
        if execution_success:
            exit(0)
        else:
            exit(1)
