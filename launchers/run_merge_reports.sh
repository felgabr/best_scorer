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
export PY_MERGE_FS=$PROD_ROUTE/A_merge_feat_selec.py
export PY_MERGE_PF=$PROD_ROUTE/B_merge_performance.py
export JOBNAME_1=merge_feat_selection
export JOBNAME_2=merge_performance


log "SH_INFO: Running $PY_MERGE_FS"
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
        $PY_MERGE_FS 2> error1.log &
        
log "SH_INFO: Running $PY_MERGE_PF"
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
        $PY_MERGE_PF 2> error2.log &

ret_code=$?
if [[ ${ret_code} != 0 ]]; then
   log "SH_ERROR:($ret_code) while executing 0_run_merge_reports.sh"
   exit ${ret_code}
fi
log "SH_INFO: Finished OK"

exit 0