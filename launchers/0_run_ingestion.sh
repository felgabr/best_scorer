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
export PY_SPINE=$PROD_ROUTE/0_spine.py
export PY_CLEAN=$PROD_ROUTE/1_clean.py
export JOBNAME_1=spine
export JOBNAME_2=clean_A

log "SH_INFO: Running $PY_SPINE"
nohup spark-submit \
        --name $JOBNAME_1 \
        --py-files /project.egg,$PROD_ROUTE/scorer.zip \
        --keytab /home/ad/$USER/$USER.keytab \
        --principal $USER@DOMAIN \
        --master yarn \
        --conf /bin/python3.7 \
        --conf spark.driver.memory=24g \
        --deploy-mode cluster \
        --files $PROD_ROUTE/config.json,$PROD_ROUTE/logging.conf, \
        $PY_SPINE 2> error1.log &

sleep 5m
applicationId=$(yarn application -list -appStates RUNNING | awk -v tmpJob=$JOBNAME_1 '{ if( $2 == tmpJob) print $1 }')
while [ ! -z $applicationId ]
do
sleep 5m
log "Job: ${JOBNAME_1} is already running. ApplicationId: ${applicationId}"
applicationId=$(yarn application -list -appStates RUNNING | awk -v tmpJob=$JOBNAME_1 '{ if( $2 == tmpJob) print $1 }')
done

log "SH_INFO: Running $PY_CLEAN"
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
        $PY_CLEAN 2> error2.log &

ret_code=$?
if [[ ${ret_code} != 0 ]]; then
   log "SH_ERROR:($ret_code) while executing 0_run_ingestion.sh"
   exit ${ret_code}
fi
log "SH_INFO: Finished OK"

exit 0