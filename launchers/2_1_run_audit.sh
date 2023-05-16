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
export PY_AUDITNUM=$PROD_ROUTE/41_audit_num.py
export PY_AUDITCAT=$PROD_ROUTE/41_audit_cat.py
export JOBNAME_1=auditnum_train
export JOBNAME_2=auditnum_test
export JOBNAME_3=auditnum_val
export JOBNAME_4=auditcat_train
export JOBNAME_5=auditcat_test
export JOBNAME_6=auditcat_val


log "SH_INFO: Running $PY_AUDITNUM"
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
        $PY_AUDITNUM 2> error1.log &

log "SH_INFO: Running $PY_AUDITNUM"
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
        $PY_AUDITNUM --split test 2> error2.log &

log "SH_INFO: Running $PY_AUDITNUM"
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
        $PY_AUDITNUM --split val 2> error3.log &

log "SH_INFO: Running $PY_AUDITCAT"
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
        $PY_AUDITCAT 2> error4.log &

log "SH_INFO: Running $PY_AUDITCAT"
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
        $PY_AUDITCAT --split test 2> error5.log &

log "SH_INFO: Running $PY_AUDITCAT"
nohup spark-submit \
        --name $JOBNAME_6 \
        --py-files /project.egg,$PROD_ROUTE/scorer.zip \
        --keytab /home/ad/$USER/$USER.ktab \
        --principal $USER@DOMAIN \
        --master yarn \
        --conf /bin/python3.7 \
        --conf spark.driver.memory=24g \
        --deploy-mode cluster \
        --files $PROD_ROUTE/config.json,$PROD_ROUTE/logging.conf, \
        $PY_AUDITCAT --split val 2> error6.log &

ret_code=$?
if [[ ${ret_code} != 0 ]]; then
   log "SH_ERROR:($ret_code) while executing 2_1_run_audit.sh"
   exit ${ret_code}
fi
log "SH_INFO: Finished OK"

exit 0