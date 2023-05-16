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
export PY_TRAIN=$PROD_ROUTE/4_trainpred.py
export PY_SCORE=$PROD_ROUTE/5_score.py
export PY_PERF=$PROD_ROUTE/6_assess_perf.py
export JOBNAME_1=train
export JOBNAME_2=score
export JOBNAME_3=performance

log "SH_INFO: Running $PY_TRAIN"
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
        $PY_TRAIN 2> error1.log &

sleep 5m
applicationId=$(yarn application -list -appStates RUNNING | awk -v tmpJob=$JOBNAME_1 '{ if( $2 == tmpJob) print $1 }')
while [ ! -z $applicationId ]
do
sleep 5m
log "Job: ${JOBNAME_1} is already running. ApplicationId: ${applicationId}"
applicationId=$(yarn application -list -appStates RUNNING | awk -v tmpJob=$JOBNAME_1 '{ if( $2 == tmpJob) print $1 }')
done

log "SH_INFO: Running $PY_SCORE"
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
        $PY_SCORE 2> error2.log &

sleep 5m
applicationId=$(yarn application -list -appStates RUNNING | awk -v tmpJob=$JOBNAME_2 '{ if( $2 == tmpJob) print $1 }')
while [ ! -z $applicationId ]
do
sleep 5m
log "Job: ${JOBNAME_1} is already running. ApplicationId: ${applicationId}"
applicationId=$(yarn application -list -appStates RUNNING | awk -v tmpJob=$JOBNAME_2 '{ if( $2 == tmpJob) print $1 }')
done

log "SH_INFO: Running $PY_PERF"
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
        $PY_PERF 2> error3.log &

ret_code=$?
if [[ ${ret_code} != 0 ]]; then
   log "SH_ERROR:($ret_code) while executing 2_run_score.sh"
   exit ${ret_code}
fi
log "SH_INFO: Finished OK"

exit 0