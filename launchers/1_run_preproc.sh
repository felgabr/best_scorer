#!/bin/bash

log() {
    date_format='+%Y-%m-%d %H:%M:%S'
    echo "*** $(date "$date_format") $1"
}
export APPS_HOME=$HOME
# Domain
export DOMAIN=
# User name
export USER_NAME=$USER
# Keytab path
export KEYTAB_FILE=$APPS_HOME/$USER_NAME.ktab
# Kerberos kinit command
export LOGIN_KERBEROS="kinit $USER_NAME -k -t $KEYTAB_FILE"
log "SH_INFO: Authenticate user: $USER_NAME"
$LOGIN_KERBEROS
# Global Jupyter variables.
export PARCEL_ANACONDA3_PATH=
source $PARCEL_ANACONDA3_PATH/scripts/env.sh
# job files path
export PROD_ROUTE=/home/ad/$USER_NAME/ProjectName
export PY_MERGE=$PROD_ROUTE/2_merge.py
export PY_INFG=$PROD_ROUTE/3a_featsel_infogain.py
export PY_CHI=$PROD_ROUTE/3b_featsel_chisqrd.py
export PY_CRAM=$PROD_ROUTE/3c_featsel_cramer.py
export PY_MUT=$PROD_ROUTE/3d_featsel_mutual.py
export JOBNAME_1=merge
export JOBNAME_2=infogain
export JOBNAME_3=chisqrd
export JOBNAME_4=cramer
export JOBNAME_5=mutual

log "SH_INFO: Running $PY_MERGE"
nohup spark-submit \
        --name $JOBNAME_1 \
        --py-files /project.egg,$PROD_ROUTE/scorer.zip \
        --keytab /home/ad/$USER/$USER.ktab \
        --principal $USER@DOMAIN \
        --master yarn \
        --conf /bin/python3.7 \
        --conf spark.driver.memory=24g \
        --deploy-mode cluster \
        --files $PROD_ROUTE/config.json,$PROD_ROUTE/logging.conf, \
        $PY_MERGE 2> error1.log &

sleep 5m
applicationId=$(yarn application -list -appStates RUNNING | awk -v tmpJob=$JOBNAME_1 '{ if( $2 == tmpJob) print $1 }')
while [ ! -z $applicationId ]
do
sleep 5m
log "Job: ${JOBNAME_1} is already running. ApplicationId: ${applicationId}"
applicationId=$(yarn application -list -appStates RUNNING | awk -v tmpJob=$JOBNAME_1 '{ if( $2 == tmpJob) print $1 }')
done

log "SH_INFO: Running $PY_INFG"
nohup spark-submit \
        --name $JOBNAME_2 \
        --py-files /project.egg,$PROD_ROUTE/scorer.zip \
        --keytab /home/ad/$USER/$USER.ktab \
        --principal $USER@DOMAIN \
        --master yarn \
        --conf /bin/python3.7 \
        --conf spark.driver.memory=24g \
        --deploy-mode cluster \
        --files $PROD_ROUTE/config.json,$PROD_ROUTE/logging.conf, \
        $PY_INFG 2> error2.log &

log "SH_INFO: Running $PY_CHI"
nohup spark-submit \
        --name $JOBNAME_3 \
        --py-files /project.egg,$PROD_ROUTE/scorer.zip \
        --keytab /home/ad/$USER/$USER.ktab \
        --principal $USER@DOMAIN \
        --master yarn \
        --conf /bin/python3.7 \
        --conf spark.driver.memory=24g \
        --deploy-mode cluster \
        --files $PROD_ROUTE/config.json,$PROD_ROUTE/logging.conf, \
        $PY_CHI 2> error3.log &

log "SH_INFO: Running $PY_CRAM"
nohup spark-submit \
        --name $JOBNAME_4 \
        --py-files /project.egg,$PROD_ROUTE/scorer.zip \
        --keytab /home/ad/$USER/$USER.ktab \
        --principal $USER@DOMAIN \
        --master yarn \
        --conf /bin/python3.7 \
        --conf spark.driver.memory=24g \
        --deploy-mode cluster \
        --files $PROD_ROUTE/config.json,$PROD_ROUTE/logging.conf, \
        $PY_CRAM 2> error4.log &

log "SH_INFO: Running $PY_MUT"
nohup spark-submit \
        --name $JOBNAME_5 \
        --py-files /project.egg,$PROD_ROUTE/scorer.zip \
        --keytab /home/ad/$USER/$USER.ktab \
        --principal $USER@DOMAIN \
        --master yarn \
        --conf /bin/python3.7 \
        --conf spark.driver.memory=24g \
        --deploy-mode cluster \
        --files $PROD_ROUTE/config.json,$PROD_ROUTE/logging.conf, \
        $PY_MUT 2> error5.log &

ret_code=$?
if [[ ${ret_code} != 0 ]]; then
   log "SH_ERROR:($ret_code) while executing 1_run_preproc.sh"
   exit ${ret_code}
fi
log "SH_INFO: Finished OK"

exit 0