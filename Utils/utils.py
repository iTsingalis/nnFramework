import os
from pathlib import Path


def check_folder(name, exp_folder):
    run = 1
    run_folder = os.path.join(exp_folder, '{}{}'.format(name, run))
    if not os.path.exists(run_folder):
        Path(run_folder).mkdir(parents=True, exist_ok=True)

        print("Path {} created".format(run_folder))
        return run_folder

    while os.path.exists(run_folder):
        run += 1
        run_folder = os.path.join(exp_folder, '{}{}'.format(name, run))
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    print("Path {} created".format(run_folder))
    return run_folder
