""" Module doctring. """
import os

CONFIG = {
    "version": "0.3.1",
    "codename": "RFE_2p",
    "datasets": ["christine", "jasmine", "madeline", "philippine", "sylvine"],
    "random_seed": 1,
    "n_jobs": -1,
    "zipme": True,
    "root_dir": os.path.abspath("."),
    "running_on_codalab": False
}

CONFIG['default_output_dir'] = os.path.join(CONFIG['root_dir'], '..', 'output_' + CONFIG['codename'])

codalab_run_dir = os.path.join(CONFIG['root_dir'], "program")
if os.path.isdir(codalab_run_dir):
    CONFIG['root_dir'] = codalab_run_dir
    CONFIG['running_on_codalab'] = True
    print "Running on Codalab!"
