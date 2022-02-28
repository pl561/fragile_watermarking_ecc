from __future__ import print_function
import os

HOME = os.environ["HOME"]
import argparse
import sys
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import cv2
from sage.all import *

from ecc import sqrt_int

from elcodes import ErrorLocatingCodeSp2
from embeddings import TamperingLocalizationLSBSP,\
    TamperingLocalizationQIMDWT
from embedding_qi2015singular import TamperingLocalization_qi2015singular
from authentication_functions import auth_modes

from wavelets import band_codes, wt_types, wt_modes

from attacks import tamper_image, apply_tamper_pattern
from attacks import image_processings
import metrics as mtc
from imagedatabase import get_tampered_realistic_png
from imagedatabase import get_tampered_realistic_png_resized
from loggingtools import setup_logger
from pathtools import EPath
from image import stack_images, text2image
from timetools import print_time, TimeTicker
import pysnooper
# https://github.com/cool-RR/PySnooper
import latex

# lena_fname = os.path.join(HOME, "Images/lena.png")
# lena_basename = os.path.basename(lena_fname)

# dpp0022_x_fname = os.path.join(HOME, "Images/DPP0022_x.png")
# dpp0022_t_fname = os.path.join(HOME, "Images/DPP0022_t.png")

# https://stackoverflow.com/questions/42865805/add-a-row-with-means-of-columns-to-pandas-dataframe

### CLEANING ###
# block_config is not used anymore
### 10 images
# pysage fwecc.py --nbimage=10 -n9 -s9 --mode=SYN
# pysage fwecc.py --nbimage=10 -n9 -s9 --mode=LOC
# pysage fwecc.py --nbimage=10 -n9 -s9 --mode=DEC

# pysage fwecc.py --nbimage=10 -n16 -s16 --mode=SYN
# pysage fwecc.py --nbimage=10 -n16 -s16 --mode=LOC
# pysage fwecc.py --nbimage=10 -n16 -s16 --mode=DEC

### 50 images
# pysage fwecc.py --nbimage=50 -n9 -s9 --mode=SYN
# pysage fwecc.py --nbimage=50 -n9 -s9 --mode=LOC
# pysage fwecc.py --nbimage=50 -n9 -s9 --mode=DEC

# pysage fwecc.py --nbimage=50 -n16 -s16 --mode=SYN
# pysage fwecc.py --nbimage=50 -n16 -s16 --mode=LOC
# pysage fwecc.py --nbimage=50 -n16 -s16 --mode=DEC

versions = {
    1  : "SP LSB RS(8, 1, 3) 1CW8L8C, 1S1L8C LOC",

    2  : "SP LSB RS(8, 1, 3) 1CW8L8C, 1S2L4C LOC",
    3  : "SP LSB RS(8, 1, 3) 1CW8L8C, 1S2L4C DEC",
    4  : "SP LSB RS(8, 1, 3) 1CW8L8C, 1S2L4C SDET",

    5  : "SP LSB RS(9, 1) 1CW9L9C, 1S3L3C SDET",
    6  : "SP LSB RS(9, 1) 1CW9L9C, 1S3L3C LOC",
    7  : "SP LSB RS(9, 1) 1CW9L9C, 1S3L3C DEC",
    # n0^2 = n, s0^2 = s, s extension degree --> GF_{q^s}
    10 : "SP LSB RS(n, 1) 1CW n0*s0 L n0*s0 C, 1S s0 L s0 C SYN|LOC|DET",

}


def benchmark(nb_image, downsizedcanon,
              code_params, qimdwt_params, attack_params,
              method, auth_mode,
              tmp_dir=None,
              bin_dir=None,
              small_image=False,
              print_end_results=False):

    ver = "v{}".format(10)
    # experiment_dir = EPath(HOME).join("tmp/fwecc") # dir to store results
    if tmp_dir is None:
        tmp_dir = EPath(HOME).join("tmp")
    else:
        tmp_dir = EPath(tmp_dir).join("tmp")
        tmp_dir.mkdir()

    if bin_dir is None:
        bin_dir = EPath(HOME).join("data_bin")
    else:
        bin_dir = EPath(bin_dir)
        bin_dir.mkdir()

    exp_dir = tmp_dir.join("fwecc")  # dir to store results
    n, s = code_params
    prefix = "{}_{}_{}_{}".format(n, s, method, auth_mode)
    if method == "QIMDWT":
        info = "_".join(map(str, qimdwt_params))
    elif method == "LSBSP":
        info = "_".join(map(str, method))
    elif method == "QISVD":
        info = "_".join(map(str, method))
    else:
        raise ValueError("Unknown embedding")
    prefix = "_".join([prefix, info])


    a_name, a_val = attack_params
    if a_name is not None:
        attack_info = "_".join(map(str, attack_params))
        prefix = "_".join([prefix, attack_info])
    # "bugs" with dots in EPath, or incompatibility
    prefix = prefix.replace(".", "f")
    cmd_line = " ".join(sys.argv)

    mylogger = setup_logger(cmd_line, str(tmp_dir.join("fwecc.log")),
                            use_console_handler=True)
    mylogger.info("Starting benchmark")

    exp_dir = exp_dir.add_after_stem(prefix)
    exp_dir.mkdir()
    exp_data = exp_dir.join("data")
    exp_data.mkdir()
    data_csv = exp_data.join("csv")
    data_csv.mkdir()
    data_tex = exp_data.join("tex")
    data_tex.mkdir()

    exp_images = exp_dir.join("images")
    exp_images.mkdir()

    project_root = EPath(HOME).join("research/fragile_watermarking")

    image_dir = project_root.join("images").join(ver.upper())
    data_dir = project_root.join("data").join(ver.upper())
    document_dir = project_root.join("document")

    # can represetned csv or tex when saving file
    bsn_allre = EPath("table_allresults").replace_parents(exp_data)
    bsn_allre = bsn_allre.add_before_stem(prefix)
    bsn_allre = bsn_allre.add_after_stem(ver)

    # allresults_ep_csv = bsn_allresults
    # allresults_ep_tex = bsn_allresults.replace_suffix(".tex")
    # print(allresults_ep_tex)
    # inside function, write results with append mode, so clear file first

    exp_dir.join("commandline").write(cmd_line)

    results_ep = EPath("results_1img.tex").add_after_stem(ver)
    results_ep = results_ep.add_before_stem(prefix)
    results_ep = results_ep.replace_parents(data_tex)
    results_ep.removefile()
    print(results_ep)
    results_ep.touch()



    dfs = []

    if downsizedcanon:
        mydataset = get_tampered_realistic_png_resized()

        print("Using downsized Canon dataset.")
    else:
        mydataset = get_tampered_realistic_png()
        print("Using full size Canon dataset")

    fname_pairs = zip(*mydataset)[: nb_image]
    assert len(fname_pairs) != 0, "No image database !"
    time.sleep(1)
    i = 0
    for fname_x, fname_t in fname_pairs:
        epx, ept = EPath(fname_x), EPath(fname_t)
        img_num = "{}/{}".format(str(i).zfill(2), nb_image)
        print(img_num, epx.stem())
        result = authentication_scenario(epx, ept, ver, exp_dir, exp_data,
                                         exp_images, image_dir, document_dir,
                                         results_ep, 
                                         code_params, qimdwt_params,
                                         attack_params,
                                         method, auth_mode,
                                         small_image=small_image)
        dfs.append(result.set_index("File name"))

        concatenated = pd.concat(dfs)
        latex_allresults = concatenated.to_latex()

        #### do not use allresults_ep_csv and allresults_ep_tex
        # concatenated.to_csv(str(allresults_ep_csv), sep=';')
        # allresults_ep_tex.write_tex(latex_allresults)

        bsn_allre.replace_parents(data_csv).writedf_tocsv(concatenated)
        bsn_allre.replace_parents(data_tex).write_tex(latex_allresults)

        i += 1

    concatenated = pd.concat(dfs)
    concatenated.loc['mean'] = concatenated.mean()
    concatenated.loc['std dev'] = concatenated.std()

    int_headers = ["TP", "FP", "FN", "TN",
                   "nb cw", "nb cw error", "nb cw controlled", "p", "s"]

    # concatenated = concatenated.round(3)
    for h, t in zip(concatenated.columns, concatenated.dtypes):
        if h in int_headers:
            concatenated[h] = concatenated[h].astype(np.int)

        else:
            if t == np.float:
                concatenated[h] = concatenated[h].round(4)

    latex_allresults = concatenated.to_latex()

    # allresults_ep_tex.write_tex(latex_allresults)
    # allresults_ep_csv.writedf_tocsv(concatenated)

    bsn_allre.replace_parents(data_tex).write_tex(latex_allresults)
    bsn_allre.replace_parents(data_csv).writedf_tocsv(concatenated)


    print("\n*** DATAFRAME ***\n")
    print(concatenated.to_string(justify="right"))

    # , "File name" est le nom de la colone d'index
    latex_groups = {
        "article"          : ["F1 score", "PPV (prec)",
                              "FPR (false alarm)", "FNR (miss detect)"],

        "general"          : ["Code", "Version", "Image size",
                              "p", "s", "C(n, k)"],
        "time"             : ["t_embed", "t_tamper", "t_auth"],
        "code"             : ["nb cw", "nb cw error", "nb cw controlled",
                              "control rate", "control rate on error"],
        "quality"          : ["PSNR", "MSE"],
        # "quality2"         : ["qim delta", "PSNR"],
        # "quality_all"          : ["qim delta", "PSNR", "SSIM", "MSE", "dwt level", "dwt band"],
        "confusionmatrix"  : ["TP", "FP", "FN", "TN"],
        "metrics1"         : ["F1 score", "MCC",
                              "Accuracy", "TPR (recall)", "PPV (prec)"],
        "metrics2"         : ["PPV (prec)", "FNR (miss detect)",
                              "FPR (false alarm)", "FDR (false disc.)",
                              "FOR (false omis.)"],

    }

    print("=" * 50)
    for theme, header in latex_groups.items():
        subtable = concatenated[header]
        theme = theme.replace(' ', '_')
        ep = bsn_allre.add_after_stem(theme)
        # print()
        # print(ep)

        tex_fname = ep.replace_parents(data_tex)
        csv_fname = ep.replace_parents(data_csv)

        # print(data_tex)
        # print(tex_fname)
        tex_fname.write_tex(subtable.to_latex())
        csv_fname.writedf_tocsv(subtable)

        if theme == "article":

            # print(tex_fname.add_suffix(".tex"))
            tex_fname.add_suffix(".tex").copyto(bin_dir)
            csv_fname.add_suffix(".csv").copyto(bin_dir)

        # print(ep, ep.exists())
        if print_end_results:
            print(theme)
            print(subtable)

    mylogger.info("Stopping benchmark")


def authentication_scenario(fname_x, fname_t, version, exp_dir, exp_data,
                            exp_images, image_dir, document_dir, results_ep,
                            code_params, qimdwt_params, attack_params,
                            method, auth_mode, small_image=False):
    apply_tamper         = True # False
    apply_authentication = True # False

    out_fname_x = fname_x.replace_parents(exp_images)
    prefix = exp_dir[-1].string()
    out_fname_x = out_fname_x.add_before_stem(prefix)
    # print("small image ", small_image)
    img_x = fname_x.imread_gs_int(small_image=small_image)
    img_t = fname_t.imread_gs_int(small_image=small_image)

    dflabels = OrderedDict()

    n, s = code_params
    p = 2
    # s = 9
    q = p ** s
    # n = 9
    k = 1
    # field = GF(q, 'a')
    # eval_elts = field.list()[1:n+1]
    # code = codes.GeneralizedReedSolomonCode(eval_elts, k)

    # ECC

    # tampering localization algorithm
    if method == "LSBSP":
        code = ErrorLocatingCodeSp2(s, n)
        tl_inst = TamperingLocalizationLSBSP(code, auth_mode)
    elif method == "QIMDWT":
        code = ErrorLocatingCodeSp2(s, n)
        delta, level, dlevel, band_name, wt_type, wt_mode = qimdwt_params
        tl_inst = TamperingLocalizationQIMDWT(code, auth_mode,
                                              delta, level, 
                                              dlevel, band_name,
                                              wt_type=wt_type,
                                              wt_mode=wt_mode)
    elif method == "QISVD":
        tl_inst = TamperingLocalization_qi2015singular(img_x)
    else:
        tl_inst = None
        raise ValueError("Unknown method")

    ticker = TimeTicker()
    ### EMBEDDING

    print("{}_Embedding...".format(auth_mode))
    ticker.tick()
    img_y = tl_inst.embedding(img_x)
    t_embed = ticker.tick()
    ticker.print_time("Embedding time            ")

    img_diff = np.abs(img_x - img_y)

    print()
    print("{}_Tamper...".format(auth_mode))
    ### TAMPER
    ticker.tick()
    if apply_tamper:
        img_z, tamper_map_x = apply_tamper_pattern(img_y, img_x, img_t)
    else:
        img_z, tamper_map_x = img_y, np.zeros_like(img_y)
    t_tamper = ticker.tick()
    ticker.print_time("Tamper time               ")
    # dflabels["Tamper time"] = (end-start)*1000


    # IMAGE PROCESSING AFTER TAMPERING
    a_name, a_val = attack_params
    if a_name is not None:
        print("{}_Attack...".format(auth_mode))
        a_func = image_processings[a_name]
        ticker.tick()
        img_z = a_func(img_z, a_val)
        t_attack = ticker.tick()
        ticker.print_time("Attack time              ")

    ### DETECTION WITH TAMPER
    print("{}_Authentication...".format(auth_mode))
    ticker.tick()
    if apply_authentication:
        tamper_map_z, cw_stats = tl_inst.detection(img_z, img_y)
    else:
        tamper_map_z, cw_stats = tamper_map_x, (0, 0, 0)
    t_auth = ticker.tick()
    nb_cw, nb_cw_error, nb_cw_controlled = cw_stats
    ticker.print_time("Authentication time      ")

    img_confmap = mtc.confusion_map(tamper_map_x, tamper_map_z)

    marked_fname = out_fname_x.add_after_stem("marked")
    tampered_fname = out_fname_x.add_after_stem("tampered")
    tm_x_fname = out_fname_x.add_after_stem("tm_x")
    tm_z_fname = out_fname_x.add_after_stem("tm_z")
    confmap_fname = out_fname_x.add_after_stem("confmap")
    # confmap_fname = out_fname_x.add_after_stem("confmap_cropped")

    # marked_fname.imwrite(img_y)
    # tampered_fname.imwrite(img_z)
    # tm_x_fname.imwrite(tamper_map_x)
    # tm_z_fname.imwrite(tamper_map_z)
    # confmap_fname.imwrite(img_confmap)
    # confmap_fname.imwrite(mtc.autocrop_confusionmap(img_confmap))

    ## TAG1
    ## goto TAG2 to see image writing


    conf_mat = mtc.confusion_matrix(tamper_map_x, tamper_map_z)
    print("Confusion matrix :")
    print(conf_mat)



    if nb_cw_error == 0:
        nb_cw_error = float(nb_cw_error) + 1e-6
    nb_cw = float(nb_cw)
    nb_cw_error = float(nb_cw_error)

    dflabels["Code"]         = __file__
    dflabels["Version"]      = version
    dflabels["File name"]    = fname_x.stem_noparam().string()
    dflabels["Image size"]   = "{}x{}".format(*img_x.shape)
    dflabels["p"]            = p
    dflabels["s"]            = s
    dflabels["C(n, k)"]      = n, k
    dflabels["embedding"]    = method
    dflabels["auth mode"]    = auth_mode

    if method == "QIMDWT":
        delta, level, dlevel, band_name, wt_type, wt_mode = qimdwt_params
        dflabels["qim delta"]    = delta
        dflabels["dwt level"]    = level
        dflabels["dwt dlevel"]   = dlevel
        dflabels["dwt band"]     = band_name
        dflabels["dwt type"]     = wt_type
        dflabels["dwt mode"]     = wt_mode

    if a_name is not None:
        dflabels["attack name"]      = a_name
        dflabels["attack value"]     = a_val

    dflabels["t_embed"]      = t_embed
    dflabels["t_tamper"]     = t_tamper
    dflabels["t_auth"]       = t_auth

    dflabels["nb cw"]                   = nb_cw
    dflabels["nb cw error"]             = nb_cw_error
    dflabels["nb cw controlled"]        = nb_cw_controlled
    dflabels["control rate"]            = 0 # nb_cw_controlled/nb_cw
    dflabels["control rate on error"]   = 0 # nb_cw_controlled/nb_cw_error

    dflabels["PSNR"]       = mtc.compute_psnr(img_x, img_y)
    dflabels["MSE"]        = mtc.compute_mse(img_x, img_y)

    tp, fp, fn, tn = conf_mat.astype(np.float64).flatten()
    dflabels["TP"]         = tp
    dflabels["FP"]         = fp
    dflabels["FN"]         = fn
    dflabels["TN"]         = tn

    dflabels["F1 score"]          = mtc.compute_f1(conf_mat)
    dflabels["Accuracy"]          = mtc.compute_acc(conf_mat)
    dflabels["TPR (recall)"]             = mtc.compute_tpr(conf_mat)
    dflabels["MCC"]               = mtc.compute_mcc(conf_mat)

    dflabels["PPV (prec)"]        = mtc.compute_ppv(conf_mat)
    dflabels["FPR (false alarm)"]        = mtc.compute_fpr(conf_mat)
    dflabels["FNR (miss detect)"]        = mtc.compute_fnr(conf_mat)

    dflabels["FDR (false disc.)"]        = mtc.compute_fdr(conf_mat)
    dflabels["FOR (false omis.)"]        = mtc.compute_for(conf_mat)

    df_measures = pd.DataFrame()
    for key, value in dflabels.items():
        df_measures[key] = [value]

    dir_img_document = EPath("../images").join(version.upper()).string()
    fig_marked = latex.image_figure([marked_fname],
                                    params="[!h]",
                                    caption=marked_fname.stem_noparam(),
                                    width="3.8cm",
                                    parent=dir_img_document)
    fig_tampered = latex.image_figure([tampered_fname],
                                      params="[!h]",
                                      caption=tampered_fname.stem_noparam(),
                                      width="3.8cm",
                                      parent=dir_img_document)
    fig_confmap = latex.image_figure([confmap_fname],
                                     params="[!h]",
                                     caption=confmap_fname.stem_noparam(),
                                     width="3.8cm",
                                     parent=dir_img_document)

    caption_fname = marked_fname.stem_noparam().add_after_stem(version)
    caption = "{}. From left to right. Watermarked image. Tampered image. Confusion map.".format(
        caption_fname)

    fig_all = latex.image_figure([marked_fname,
                                  tampered_fname,
                                  confmap_fname],
                                 params="[!h]",
                                 caption=caption,
                                 width="3.8cm",
                                 parent=dir_img_document)

    table_measures = df_measures.T.to_latex(header=False)
    latex_confmat = pd.DataFrame(conf_mat)
    table_confmat = latex.tolatex_confmat(conf_mat)

    # results_ep = EPath("results_1img_v1.tex").replace_parents(experiment_dir)

    latex_results = [fig_all, fig_marked, fig_tampered, fig_confmap,
                     table_measures, table_confmat, "%" * 40]
    latex_content = "\n\n".join(latex_results) + "\n\n"


    results_ep.write_tex(latex_content, mode='a')
    # with open(str(results_ep), "a") as fd:
    #     fd.write(latex_content)
        # results_ep.copyto(document_dir)

        # write images in article image directory
        # marked_fname.replace_parents(image_dir).imwrite(img_y)
        # tampered_fname.replace_parents(image_dir).imwrite(img_z)
        # confmap_fname.replace_parents(image_dir).imwrite(img_confmap)

    measures_text = df_measures.T.to_string(justify="right", header=False)
    print(measures_text)

    text_image = text2image(measures_text, img_y.shape)
    print(text_image.shape)
    ## TAG2
    ## goto TAG1 comment to see fnames
    images = img_y, img_z, tamper_map_x, tamper_map_z, img_confmap, text_image

    img_all = stack_images((2, 3), *images)
    all_fname = out_fname_x.add_after_stem("all")
    all_fname.imwrite(img_all)



    return df_measures



# @pysnooper.snoop()
def main():
    parser = argparse.ArgumentParser(
        description="Fragile watermarking and error control codes")

    parser.add_argument("-n", "--codelength",
                        type=int, required=True,
                        help="code length")
    parser.add_argument("-s", "--extensiondegree",
                        type=int, required=True,
                        help="symbol binary size/extension degree")
    parser.add_argument("--authmode",
                        type=str, required=True,
                        choices=list(auth_modes.keys()),
                        help="auth mode")

    parser.add_argument("--embedding",
                        default="LSBSP", choices=["LSBSP", "QIMDWT", "QISVD"],
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

    # other parameters
    parser.add_argument("--nbimage", type=int, required=True,
                        help="number of image")
    parser.add_argument("--smallimage",
                        default=False, action='store_true',
                        help="Use smaller images for faster algorithm, "
                             "useful for debug/test...")
    parser.add_argument("--downsizedcanon",
                        default=False, action='store_true',
                        help="Use downsized Canon 60D dataset")


    parser.add_argument("--printendresults",
                        action='store_true',
                        help="Print all the results at the end")
    parser.add_argument("--tmpdir", type=str,
                        default=None,
                        help="Bin directory to store results")
    parser.add_argument("--bindir", type=str,
                        default=None,
                        help="Bin directory to store results")

    args = parser.parse_args()
    nbimage = args.nbimage
    n = args.codelength
    s = args.extensiondegree
    embedding = args.embedding.upper()
    auth_mode = args.authmode.upper()

    print("n            = ", n)
    print("s            = ", s)
    print("method       :", embedding)
    print("auth mode    :", auth_mode)
    print("nbimage      =", nbimage)
    print("imgprocessing=", args.attackname)
    print("small img    =", args.smallimage)
    # return

    code_params = n, s
    qimdwt_params = args.delta, \
                    args.level, args.dlevel, args.bandname, \
                    args.wttype, args.wtmode
    attack_params = args.attackname, args.attackvalue


    benchmark(nbimage, args.downsizedcanon,
              code_params, qimdwt_params, attack_params,
              embedding, auth_mode,
              bin_dir=args.bindir, tmp_dir=args.tmpdir,
              print_end_results=args.printendresults,
              small_image=args.smallimage)


if __name__ == "__main__":
    sys.exit(main())
