import run_blr_ais
import sys
import numpy as np
from absl import flags


def exp():
    FLAGS = flags.FLAGS
    res = {}
    for dataset in ["yacht","energy-efficiency"]:
        res[dataset] = {}
        for reverse in [False, True]:
            res_temp=[]
            for i in range(10):
                flags.FLAGS.__setattr__("dataset_name", dataset)
                flags.FLAGS.__setattr__("reverse_value", reverse)
                flags.FLAGS.__setattr__("split_seed", i)
                sys.argv = ["run_blr_ais.py"]
                flags.FLAGS(sys.argv)
                res_temp.append(run_blr_ais.main(sys.argv))
            res[dataset][str(reverse)] = (np.mean(res_temp, axis=0), np.std(res_temp, axis=0))
    print(res)
    

if __name__ == "__main__":
    exp()