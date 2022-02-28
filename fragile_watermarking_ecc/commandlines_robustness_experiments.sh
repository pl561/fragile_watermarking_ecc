#!/usr/bin/env bash


#pysage fwecc.py --downsizedcanon -n4 -s4 --authmode=ELLOC --embedding=QIMDWT --level=2 --dlevel=1 --bandname=hh --wttype=haar --bindir=/home/plefevre/tmp/data_robustness_elloc --tmpdir=/mnt/part278 --delta=30 --nbimage=50




mypython=$HOME/sagemath/SageMath/local/bin/python

lvl=2
nbimg=50
nn=4
ss=4
dlt=30
d=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}_${dlt}_jpeg
tmpd=/mnt/part278

for av in `seq 44 1 99`
do
    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --delta=${dlt} --attackname=compression --attackvalue=${av}
done


#(py3.6) plefevre:data_can_4_4_2_60/ $ grep mean  *.tex                                                                                                                                   [18:01:36]
#4_4_QIMDWT_ELLOC_60_2_1_jpeg_80      & 0.4058 &  0.4814 & 0.0166 & 0.5200 \\
#4_4_QIMDWT_ELLOC_60_2_1_jpeg_84      & 0.4379 &  0.6204 & 0.0090 & 0.5424 \\
#4_4_QIMDWT_ELLOC_60_2_1_jpeg_88      & 0.4541 &  0.6806 & 0.0058 & 0.5527 \\
#4_4_QIMDWT_ELLOC_60_2_1_jpeg_92      & 0.4818 &  0.7277 & 0.0045 & 0.5533 \\
#4_4_QIMDWT_ELLOC_60_2_1_jpeg_96      & 0.4765 &  0.7320 & 0.0043 & 0.5510 \\
#(py3.6) plefevre:data_can_4_4_2_60/ $ cd ..                                                                                                                                              [18:01:45]
#(py3.6) plefevre:tmp/ $ grep mean  data_can_4_4_2_45/*.tex                                                                                                                               [18:03:02]
#data_can_4_4_2_45/4_4_QIMDWT_ELLOC_45_2_1_jpeg_80      & 0.2701 &  0.2144 & 0.0780 & 0.4452 \\
#data_can_4_4_2_45/4_4_QIMDWT_ELLOC_45_2_1_jpeg_84      & 0.4085 &  0.4037 & 0.0317 & 0.4391 \\
#data_can_4_4_2_45/4_4_QIMDWT_ELLOC_45_2_1_jpeg_88      & 0.4573 &  0.5891 & 0.0139 & 0.4850 \\
#data_can_4_4_2_45/4_4_QIMDWT_ELLOC_45_2_1_jpeg_92      & 0.4810 &  0.6393 & 0.0087 & 0.5063 \\
#data_can_4_4_2_45/4_4_QIMDWT_ELLOC_45_2_1_jpeg_96      & 0.5212 &  0.7065 & 0.0059 & 0.4956 \\
#(py3.6) plefevre:tmp/ $ grep mean  data_can_4_4_2_30/*.tex                                                                                                                               [18:03:18]
#data_can_4_4_2_30/4_4_QIMDWT_ELLOC_30_2_1_jpeg_80      & 0.1582 &  0.0944 & 0.2678 & 0.3233 \\
#data_can_4_4_2_30/4_4_QIMDWT_ELLOC_30_2_1_jpeg_84      & 0.2046 &  0.1274 & 0.1889 & 0.3080 \\
#data_can_4_4_2_30/4_4_QIMDWT_ELLOC_30_2_1_jpeg_88      & 0.3323 &  0.2438 & 0.0818 & 0.3266 \\
#data_can_4_4_2_30/4_4_QIMDWT_ELLOC_30_2_1_jpeg_92      & 0.5019 &  0.5178 & 0.0263 & 0.3601 \\
#data_can_4_4_2_30/4_4_QIMDWT_ELLOC_30_2_1_jpeg_96      & 0.5534 &  0.5957 & 0.0160 & 0.3742 \\

grep mean  data_can_4_4_3_30/*.tex && grep mean  data_can_4_4_3_45/*.tex && grep mean  data_can_4_4_3_60/*.tex


#(py3.6) plefevre:tmp/ $ grep mean  data_can_4_4_3_30/*.tex; echo && grep mean  data_can_4_4_3_45/*.tex; echo && grep mean  data_can_4_4_3_60/*.tex                                       [19:00:59]
#data_can_4_4_3_30/4_4_QIMDWT_ELLOC_30_3_1_hh_haarjpeg_68 mean & 0.1462 & 0.0857 &   0.4559 &   0.1711 \\
#data_can_4_4_3_30/4_4_QIMDWT_ELLOC_30_3_1_hh_haarjpeg_72 mean & 0.1777 & 0.1071 &   0.3482 &   0.1755 \\
#data_can_4_4_3_30/4_4_QIMDWT_ELLOC_30_3_1_hh_haarjpeg_76 mean & 0.2412 & 0.1527 &   0.2199 &   0.1797 \\
#data_can_4_4_3_30/4_4_QIMDWT_ELLOC_30_3_1_hh_haarjpeg_80 mean & 0.3371 & 0.2306 &   0.1251 &   0.1867 \\
#data_can_4_4_3_30/4_4_QIMDWT_ELLOC_30_3_1_hh_haarjpeg_84 mean & 0.4866 & 0.3775 &   0.0583 &   0.1925 \\
#data_can_4_4_3_30/4_4_QIMDWT_ELLOC_30_3_1_hh_haarjpeg_88 mean & 0.6163 & 0.5320 &   0.0297 &   0.1990 \\
#data_can_4_4_3_30/4_4_QIMDWT_ELLOC_30_3_1_hh_haarjpeg_92 mean & 0.6805 & 0.6237 &   0.0207 &   0.1985 \\
#data_can_4_4_3_30/4_4_QIMDWT_ELLOC_30_3_1_hh_haarjpeg_96 mean & 0.6892 & 0.6372 &   0.0191 &   0.2036 \\
#
#data_can_4_4_3_45/4_4_QIMDWT_ELLOC_45_3_1_hh_haarjpeg_68 mean & 0.4617 & 0.3567 &   0.0622 &   0.2189 \\
#data_can_4_4_3_45/4_4_QIMDWT_ELLOC_45_3_1_hh_haarjpeg_72 mean & 0.5417 & 0.4472 &   0.0413 &   0.2187 \\
#data_can_4_4_3_45/4_4_QIMDWT_ELLOC_45_3_1_hh_haarjpeg_76 mean & 0.6103 & 0.5366 &   0.0270 &   0.2234 \\
#data_can_4_4_3_45/4_4_QIMDWT_ELLOC_45_3_1_hh_haarjpeg_80 mean & 0.6602 & 0.6080 &   0.0205 &   0.2256 \\
#data_can_4_4_3_45/4_4_QIMDWT_ELLOC_45_3_1_hh_haarjpeg_84 mean & 0.6757 & 0.6325 &   0.0184 &   0.2282 \\
#data_can_4_4_3_45/4_4_QIMDWT_ELLOC_45_3_1_hh_haarjpeg_88 mean & 0.6829 & 0.6414 &   0.0178 &   0.2279 \\
#data_can_4_4_3_45/4_4_QIMDWT_ELLOC_45_3_1_hh_haarjpeg_92 mean & 0.6851 & 0.6465 &   0.0169 &   0.2288 \\
#data_can_4_4_3_45/4_4_QIMDWT_ELLOC_45_3_1_hh_haarjpeg_96 mean & 0.6938 & 0.6545 &   0.0156 &   0.2263 \\
#
#data_can_4_4_3_60/4_4_QIMDWT_ELLOC_60_3_1_hh_haarjpeg_64 mean & 0.6185 & 0.5677 &   0.0231 &   0.2490 \\
#data_can_4_4_3_60/4_4_QIMDWT_ELLOC_60_3_1_hh_haarjpeg_68 mean & 0.6538 & 0.6157 &   0.0191 &   0.2473 \\
#data_can_4_4_3_60/4_4_QIMDWT_ELLOC_60_3_1_hh_haarjpeg_72 mean & 0.6597 & 0.6335 &   0.0175 &   0.2542 \\
#data_can_4_4_3_60/4_4_QIMDWT_ELLOC_60_3_1_hh_haarjpeg_76 mean & 0.6674 & 0.6458 &   0.0163 &   0.2625 \\
#data_can_4_4_3_60/4_4_QIMDWT_ELLOC_60_3_1_hh_haarjpeg_80 mean & 0.6742 & 0.6515 &   0.0149 &   0.2596 \\
#data_can_4_4_3_60/4_4_QIMDWT_ELLOC_60_3_1_hh_haarjpeg_84 mean & 0.6776 & 0.6548 &   0.0150 &   0.2570 \\
#data_can_4_4_3_60/4_4_QIMDWT_ELLOC_60_3_1_hh_haarjpeg_88 mean & 0.6780 & 0.6564 &   0.0148 &   0.2572 \\
#data_can_4_4_3_60/4_4_QIMDWT_ELLOC_60_3_1_hh_haarjpeg_92 mean & 0.6874 & 0.6679 &   0.0138 &   0.2548 \\
#data_can_4_4_3_60/4_4_QIMDWT_ELLOC_60_3_1_hh_haarjpeg_96 mean & 0.6938 & 0.6757 &   0.0133 &   0.2548 \\


mypython=$HOME/sagemath/SageMath/local/bin/python

lvl=3
nbimg=50
nn=4
ss=4
dlt=60
d=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}_${dlt}_jpeg
tmpd=/mnt/part278

for av in `seq 44 1 99`
do
    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --delta=${dlt} --attackname=compression --attackvalue=${av}
done

grep mean  data_can_4_4_3_60/*.csv | cut -d"_" -f17,21
grep std  data_can_4_4_3_60/*.csv | cut -d"_" -f17,21


####################################################################
## AWGN LVL3
####################################################################

mypython=$HOME/sagemath/SageMath/local/bin/python
lvl=3
nbimg=50
nn=4
ss=4
dlt=30
d=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}_${dlt}_awgn
tmpd=/mnt/part278

for av in `seq 0 1 20`
do
    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --delta=${dlt} --attackname=awgn --attackvalue=${av}
done


mypython=$HOME/sagemath/SageMath/local/bin/python
lvl=3
nbimg=50
nn=4
ss=4
dlt=45
d=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}_${dlt}_awgn
tmpd=/mnt/part278

for av in `seq 0 1 20`
do
    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --delta=${dlt} --attackname=awgn --attackvalue=${av}
done


mypython=$HOME/sagemath/SageMath/local/bin/python
lvl=3
nbimg=50
nn=4
ss=4
dlt=60
d=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}_${dlt}_awgn
tmpd=/mnt/part278

for av in `seq 0 1 20`
do
    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --delta=${dlt} --attackname=awgn --attackvalue=${av}
done

#https://stackoverflow.com/questions/6438896/sorting-data-based-on-second-column-of-a-file
grep mean  data_can_4_4_2_60_awgn/*.csv | cut -d"_" -f18,22  | sort -t"f" -k1 --numeric-sort
grep std  data_can_4_4_2_60_awgn/*.csv | cut -d"_" -f18,22  | sort -t"f" -k1 --numeric-sort

####################################################################
## AWGN LVL2
####################################################################

mypython=$HOME/sagemath/SageMath/local/bin/python
lvl=2
nbimg=50
nn=4
ss=4
dlt=30
d=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}_${dlt}_awgn
tmpd=/mnt/part278

for av in `seq 0 1 20`
do
    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --delta=${dlt} --attackname=awgn --attackvalue=${av}
done


mypython=$HOME/sagemath/SageMath/local/bin/python
lvl=2
nbimg=50
nn=4
ss=4
dlt=45
d=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}_${dlt}_awgn
tmpd=/mnt/part278

for av in `seq 0 1 20`
do
    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --delta=${dlt} --attackname=awgn --attackvalue=${av}
done


mypython=$HOME/sagemath/SageMath/local/bin/python
lvl=2
nbimg=50
nn=4
ss=4
dlt=60
d=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}_${dlt}_awgn
tmpd=/mnt/part278

for av in `seq 0 1 20`
do
    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --delta=${dlt} --attackname=awgn --attackvalue=${av}
done

grep std  data_can_4_4_3_60_awgn/*.csv | cut -d"_" -f18,22


####################################################################
## QISVD JPEG
####################################################################


mypython=$HOME/sagemath/SageMath/local/bin/python
nbimg=50
d=${HOME}/tmp/data_can_qisvd_jpeg
tmpd=/mnt/part278

for av in `seq 70 1 74`
do
    ${mypython} fwecc.py --downsizedcanon -n0 -s0 --authmode=QISVD --embedding=QISVD --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --attackname=compression --attackvalue=${av}
done

grep std  data_can_qisvd_jpeg/*.csv
# | cut -d"_" -f18,22

####################################################################
## QISVD AWGN
####################################################################

mypython=$HOME/sagemath/SageMath/local/bin/python
nbimg=50
d=${HOME}/tmp/data_can_qisvd_awgn
tmpd=/mnt/part278

for av in `seq 5 1 9`
do
    ${mypython} fwecc.py --downsizedcanon -n0 -s0 --authmode=QISVD --embedding=QISVD --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --attackname=awgn --attackvalue=0.${av}
done

grep std  data_can_qisvd_awgn/*.csv | cut -d"_" -f18,22


####################################################################
## ELLOC LVL3 LPF
####################################################################
mypython=$HOME/sagemath/SageMath/local/bin/python

lvl=3
nbimg=50
nn=4
ss=4
dlt=60
d=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}_${dlt}_lpf
tmpd=/mnt/part278

for av in `seq 1 1 1`
do
    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --delta=${dlt} --attackname=lpf --attackvalue=${av}
done


# display basename without extension
#for n in `ls *.png`; do echo ${n%.png};done

# crop images
#for n in `ls *_all.png`; do convert $n -crop 1280x720+1280+720 ${n%.png}_confmap.png;done
#for n in `ls *_all.png`; do convert $n -crop 1280x720+0+0 ${n%.png}_orig.png;done
#for n in `ls *_all.png`; do convert $n -crop 1280x720+1280+0 ${n%.png}_tampered.png;done












