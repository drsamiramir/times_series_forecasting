from pandas import read_csv
from pathlib import Path


def load_cvs_file(file_name):
    ROOT_DIR = Path(__file__).parent.parent.parent
    series = read_csv(str(ROOT_DIR)+'/data/'+file_name, header=0, index_col=0)
    values = series.values
    return values
