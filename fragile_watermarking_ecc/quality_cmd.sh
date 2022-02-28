#! /bin/sh

mypython=$HOME/sagemath/SageMath/local/bin/python

# In fwecc.py, function authentication_scenario
# set to False the following
##    apply_tamper         = True # False
##    apply_authentication = True # False

#basic example
#lvl=2
#nbimg=10
#nn=4
#ss=4
#d=${HOME}/tmp/data_can_${nn}_${ss}
#tmpd=/mnt/part278
#for i in `seq 30 2 70`
#do
#    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d --tmpdir=$tmpd --delta=$i
#done
#
#cat $d/*.csv | grep mean | cut -f 3 -d ';'
#cat $d/*.csv | grep std | cut -f 3 -d ';'
#
#pr -mts <(cat $d/*.csv | grep mean | cut -f 3 -d ';') <(cat $d/*.csv | grep std | cut -f 3 -d ';')


# complete experiments
lvl=1
nbimg=50
nn=16
ss=16
d1=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}
tmpd=/mnt/part278
#for i in `seq 30 1 70`
#do
#    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d1 --tmpdir=$tmpd --delta=$i
#done

lvl=2
nbimg=50
nn=4
ss=4
d2=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}
tmpd=/mnt/part278
#for i in `seq 30 1 70`
#do
#    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d2 --tmpdir=$tmpd --delta=$i
#done

lvl=2
nbimg=50
nn=9
ss=9
d3=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}
tmpd=/mnt/part278
#for i in `seq 30 1 70`
#do
#    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d3 --tmpdir=$tmpd --delta=$i
#done

lvl=2
nbimg=50
nn=16
ss=16
d4=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}
tmpd=/mnt/part278
#for i in `seq 30 1 70`
#do
#    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d4 --tmpdir=$tmpd --delta=$i
#done

lvl=3
nbimg=50
nn=16
ss=16
d5=${HOME}/tmp/data_can_${nn}_${ss}_${lvl}
tmpd=/mnt/part278
#for i in `seq 30 1 70`
#do
#    ${mypython} fwecc.py --downsizedcanon -n${nn} -s${ss} --authmode=ELLOC --embedding=QIMDWT --level=${lvl} --dlevel=1 --bandname=hh --wttype=haar --nbimage=$nbimg --bindir=$d5 --tmpdir=$tmpd --delta=$i
#done

echo $d1
echo $d2
echo $d3
echo $d4
echo $d5
echo "mean"

pr -mts <(cat $d1/*.csv | grep mean | cut -f 2 -d ';') <(cat $d1/*.csv | grep mean | cut -f 3 -d ';') <(cat $d2/*.csv | grep mean | cut -f 3 -d ';') <(cat $d3/*.csv | grep mean | cut -f 3 -d ';') <(cat $d4/*.csv | grep mean | cut -f 3 -d ';') <(cat $d5/*.csv | grep mean | cut -f 3 -d ';')

echo -e "\n"
echo "std"

pr -mts <(cat $d1/*.csv | grep std | cut -f 2 -d ';') <(cat $d1/*.csv | grep std | cut -f 3 -d ';') <(cat $d2/*.csv | grep std | cut -f 3 -d ';') <(cat $d3/*.csv | grep std | cut -f 3 -d ';') <(cat $d4/*.csv | grep std | cut -f 3 -d ';') <(cat $d5/*.csv | grep std | cut -f 3 -d ';')