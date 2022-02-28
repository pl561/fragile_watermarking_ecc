from __future__ import print_function
import os
HOME = os.environ["HOME"]

import glob
import sys
import argparse
import shlex
from subprocess import Popen, PIPE, check_output, check_call
from collections import OrderedDict
from itertools import product
import json
import pandas as pd
from pathtools import EPath
from bchcodes import bch_params
from wavelets import band_codes, wt_types, wt_modes
from attacks import image_processings
from authentication_functions import auth_modes

# n x s

auth_variants = {
    "RSDEC"      : "RS-DEC ",
    "RSLOC"      : "RS-LOC ",

    "ELLOC"      : "EL-LOC ",
    "ELLOCNOSYN" : "EL-LOC-NOSYN",

    "BCHDEC"     : "BCH-DEC ",

    # "fwecc\_"
}

def noindent():
    print("\\noindent")

def skip(size="small", ni=True):
    assert size in ["small", "med", "big"]
    print("\\{}skip".format(size))
    if ni:
        noindent()


def get_bchparams(n, s):
    n_bch, k_bch, t_bch = bch_params[(n, s)]
    string = "$n \\times s = {}, BCH({}, {}, {})$".format(n*s, n_bch, k_bch, t_bch)
    return string

def print_bchparams(n, s):
    string = get_bchparams(n, s)
    print(string)


def latex_command(cmd_name, arg):
    ret = "".join(["\\", cmd_name , "{", arg, "}"])
    return ret

def subsection(arg):
    ret = latex_command("subsection", arg)
    return ret



def extract_field_from_csv(n, s, csv_pattern, fields, directories):
    label = "mean"
    # label = "std dev"
    csv_fnames = []
    exp_dir = EPath(HOME).join("tmp")
    for d in directories:
        # build path
        pattern = exp_dir.join(d).join("data")
        gl = glob.glob(pattern.string()+"/"+csv_pattern)
        # print(d)
        # print(gl)
        if len(gl) != 0:
            csv_fnames.append(gl[0])



    data = OrderedDict()
    for d, fname in zip(directories, csv_fnames):
        df = pd.read_csv(fname, index_col=0, sep=";") # col 0 as index
        # print(df)
        # print(fname)
        data[d] = []
        for field in fields:
            data[d].append(df[field][label])

    # for item in data.items():
    #     print(item)

    print()

    results = pd.DataFrame(data, index=fields)
    # results *= 100



    latex_table = results.T.to_latex()
    for l1, l2 in auth_variants.items():
        latex_table = latex_table.replace(l1, l2)


    prefix = "fwecc\_{}\_{}\_".format(n, s)
    # print(s)
    latex_table = latex_table.replace(prefix, "")

    ns_str = "$({}, {})$".format(n, s)
    latex_table = latex_table.replace("{}", ns_str)

    print(latex_table)


def get_experiments_cmdline(n, s, modes, embedding,
                            delta,
                            level, dlevel, bandname,
                            wttype, wtmode,
                            attackname=None, attackvalue=None,
                            nb_image=10):

    interpreter = "pysage"
    filename = "fwecc.py"
    cmd_tmp = "{} {} --nbimage={} -n{} -s{} --authmode={} " \
              "--embedding={} " \
              "--delta={} " \
              "--level={} --dlevel={} --bandname={} " \
              "--wttype={} --wtmode={} " \
              "--attackname={} --attackvalue={}"


    cmdlines = []
    for mode in modes:
        cmdline = cmd_tmp.format(interpreter, filename,
                                 nb_image, n, s, mode, embedding,
                                 delta,
                                 level, dlevel, bandname,
                                 wttype, wtmode,
                                 attackname, attackvalue)
        cmdlines.append(cmdline)
    return cmdlines


def get_cmdlinescustom():
    cmdlines = []

    cl = "pysage fwecc.py --nbimage=50 --downsizedcanon -n4 -s4 --authmode=ELLOCNOSYN --embedding=QIMDWT --delta=64 --level={} --dlevel={} --bandname={} --wttype=haar --attackvalue={} --attackname=compression"

    dlvl = 1
    bn = 0
    lvl = 3

    groups = []

    for lvl in range(3, 6):
        for bn in [0, 1, 2]:
            for av in range(70, 100):
                cl_tmp = cl.format(lvl, dlvl, bn, av)
                cmdlines.append(cl_tmp)
            groups.append(cmdlines)
    return groups


def main():
    parser = argparse.ArgumentParser(description="FW+ECC cmdline generator and data processing")
    parser.add_argument("--nbimage", type=int, #required=True,
                        help="number of image")
    parser.add_argument("-n", "--codelength",
                        type=int, #required=True,
                        help="code length")
    parser.add_argument("-s", "--extensiondegree",
                        type=int, #required=True,
                        help="symbol binary size/extension degree")
    # parser.add_argument("--mode",
    #                     type=str, required=True,
    #                     choices=auth_modes,
    #                     help="auth mode")

    parser.add_argument("--embedding",
                        default="LSBSP", choices=["LSBSP", "QIMDWT"],
                        help="Representation domain of the image"
                             "for embedding")

    # QIM DWT arguments
    parser.add_argument("--delta",
                        type=int, default=None,
                        help="Number of level in DWT")

    parser.add_argument("--level",
                        type=int, default=None,
                        help="Number of level in DWT")
    parser.add_argument("--dlevel",
                        type=int, default=None,
                        help="detail level (must be <= level)")
    parser.add_argument("--bandname", type=str, default=None,
                        choices=list(band_codes.keys()),
                        help="the band extracted from DWT decomposition")
    # DWT type and mode
    parser.add_argument("--wttype", type=str, default="db4",
                        choices=wt_types,
                        help="DWT type")
    parser.add_argument("--wtmode", type=str, default="periodization",
                        choices=wt_modes,
                        help="DWT mode")

    # image processing arguments
    parser.add_argument("--attackname", type=str, default=None,
                        choices=list(image_processings.keys()),
                        help="Name of the attack for DWT embedding")
    parser.add_argument("--attackvalue", type=float, default=None,
                        help="Value for the attack")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--cmdline', action='store_true',
                       default=False,
                       help="activate cmdline option")
    group.add_argument('--dataprocessing',
                       action='store_true',
                       default=False,
                       help="process data in a dataframe")

    group.add_argument('--cmdlinefromdict', action='store_true',
                       default=False,
                       help="activate cmdline option from dict")

    parser.add_argument("--execute", action='store_true',
                       default=False,
                        help="execute every command lines")

    parser.add_argument("--cmdlinecustom", action='store_true',
                       default=False,
                        help="execute every command lines")


    args = parser.parse_args()
    nbimage = args.nbimage
    n = args.codelength
    s = args.extensiondegree
    # auth_mode = args.mode.upper()
    auth_modes_keys = tuple(auth_modes.keys())
    auth_modes_keys = [auth_mode_key.ljust(10, ' ') for auth_mode_key in auth_modes_keys]

    if args.dataprocessing:
        d0 = [
            "fwecc_{}_{}_RSDEC",
            "fwecc_{}_{}_RSLOC",

            "fwecc_{}_{}_BCHDEC",

            "fwecc_{}_{}_ELLOC",
            "fwecc_{}_{}_ELLOCNOSYN",
            ]


        csv_pattern = "*_table_allresults_*.csv"

        directories = [string.format(n, s) for string in d0]

        fields1 = [
            "PPV (prec)",
            "FNR (miss detect)",
            "FPR (false alarm)",
        ]

        fields2 = [
            "F1 score",
            "MCC",
            "Accuracy",
            "TPR (recall)",
        ]

        print(subsection(get_bchparams(n, s)))
        skip()
        extract_field_from_csv(n, s, csv_pattern, fields1, directories)

        skip()

        extract_field_from_csv(n, s, csv_pattern, fields2, directories)


        directories = [string.format(s, n) for string in d0]
        extract_field_from_csv(s, n, csv_pattern, fields1, directories)

        skip()
        extract_field_from_csv(s, n, csv_pattern, fields2, directories)


    if args.cmdline:

        get_experiments_cmdline(n, s, auth_modes_keys,
                                args.embedding,
                                args.delta,
                                args.level, args.dlevel, args.bandname,
                                args.wttype, args.wtmode,
                                args.attackname, args.attackvalue,
                                nb_image=nbimage)
        # pysage fwecc.py --nbimage=1 --mode=ELLOC --method=QIMDWT --delta=30 --level=3 --dlevel=1 --bandname=d -n9 -s9

    if args.cmdlinefromdict:
        deltas = 32, #16
        compression = 80, 90, 99
        noise = 2, 4, 6
        dlevels = 1, 2 #, 3
        bandnames = 'd',#, 'h', 'v'
        level = 3
        wttype = 'db1'
        wtmode = "periodization"
        prod_compression = product(deltas, bandnames, dlevels, compression)
        prod_noise = product(deltas, bandnames, dlevels, noise)

        all_cmdlines = []

        cpt = 1
        for delta, bandname, dlevel, av in prod_compression:
            print(cpt, n, s)
            cmdlines = get_experiments_cmdline(n, s, auth_modes_keys,
                                    args.embedding,
                                    delta,
                                    level,
                                    dlevel, bandname,
                                    wttype, wtmode,
                                      "compression", av,
                                    nb_image=nbimage)
            all_cmdlines.extend(cmdlines)
            cpt += 1
            # print(cmdlines)

        for delta, bandname, dlevel, av in prod_noise:
            print(cpt, n, s)
            cmdlines = get_experiments_cmdline(n, s, auth_modes_keys,
                                    args.embedding,
                                    delta,
                                    level,
                                    dlevel, bandname,
                                    wttype, wtmode,
                                      "awgn", av,
                                    nb_image=nbimage)
            all_cmdlines.extend(cmdlines)
            cpt += 1
            # print(cmdlines)

        # pysage dataprocessing.py --cmdlinefromdict -n9 -s4
        if args.execute:
            i = 1
            for cmd in all_cmdlines:


                if i%5 == 0:
                    # print("\n"*2)
                    print(cmd, "\n\n")
                else:
                    # print(" && ")
                    print(cmd, " && ",)
                i += 1

            # with open("/tmp/cmdlines.sh", "w") as fd:
            #     fd.write("\n".join(all_cmdlines))
            # for cmd in all_cmdlines:
            #     cmd_split = shlex.split(cmd)
                # print(cmd_split)
                # print(cmd)
                # print(os.getcwd())
                # os.system(cmd)
                # rc = check_call(cmd_split, cwd=os.getcwd(),
                #                 executable='/bin/zsh', shell=True)
                # print("return code : ", rc)

    if args.cmdlinecustom:

        groups = get_cmdlinescustom()
        for cmdlines in groups:
            joined = " && ".join(cmdlines)
            print(joined)
            print()
            print()
            print()

        # for cl in cmdlines:
        #     print(cl)




if __name__ == "__main__":
    sys.exit(main())