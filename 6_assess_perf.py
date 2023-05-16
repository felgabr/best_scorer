# ==================================================================================
# PACKAGES
# ==================================================================================
# Standard
import argparse
import datetime
import json
import logging.config
import os
import socket
import subprocess
import sys
import time
import warnings

# Third Party
import numpy as np
from pyspark.ml.classification import GBTClassificationModel, GBTClassifier
from pyspark.ml.classification import RandomForestClassifierModel, RandomForestClassifier
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
# ==================================================================================
# SETTINGS
# ==================================================================================
warnings.filterwarnings('ignore')
# ==================================================================================
# PARAMETERS pt1
# ==================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config.json')
parser.add_argument('-l', '--log', type=str, default='logging.conf')
args = parser.parse_args()
# GLOBAL
abs_path = os.getcwd()
user = os.environ['USER']
# Load config file
conf_filepath = os.path.join(abs_path, args.config)
__CONFIG__ = json.load(open(conf_filepath))
# General
project = __CONFIG__['project']
app_name = f'{project}_perfrmc'
schema = __CONFIG__['schema']
# ==================================================================================
# LOGGER
# ==================================================================================
log_filepath = os.path.join(abs_path, args.log)
logging.config.fileConfig(fname=log_filepath)
logger = logging.getLogger(project)
log_file, stdlog_file = __CONFIG__['logger_files']
# ==================================================================================
# LOCAL
# ==================================================================================
from evaluate import Evaluator
from supplementary import DateAux, LakeAux, PersistManager, ReportManager
from utils import build_spark_session, send_email
# ==================================================================================
# PARAMETERS pt2
# ==================================================================================
# Date Parameters
exec_date = (datetime.datetime.today()).strftime('%Y%m%d_%H%M%S')
# Tables
data_src = __CONFIG__['table_outputs']['score']
# Mailing
dest = __CONFIG__['from']
recp = __CONFIG__['to']
recp_cp = __CONFIG__['cc']
# Output

# ML config
config = __CONFIG__['conf']
# ==================================================================================
# MAIN
# ==================================================================================
try:
    logger.info('Running')
    logger.info('Hostname: ' + str(socket.gethostname()))

    start_time = time.time()
    # ==============================================================================
    # SPARK SESSION
    # ==============================================================================
    spark = build_spark_session(app_name, 
                                 __CONFIG__['size'], 
                                 args=[('spark.rpc.message.maxSize', 256)], 
                                 jars=[], 
                                 py_files=[__CONFIG__['py_file0'],
                                           __CONFIG__['py_file1']],
                                 path_prefix='log/')

    spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
    
    sc = spark.sparkContext

    logger.info('Spark applicationId: ' + str(sc.applicationId))
    logger.info(f'Spark Parameters: \n{spark.sparkContext.getConf().getAll()}')
    # ==============================================================================
    # LOAD DATA
    # ==============================================================================
    logger.info('Start load data')   

    data_src = LakeAux.get_most_recent_table(spark, schema, data_src)
    
    logger.info(f'Latest dataset: {data_src}')
    df = spark.read.table(f'{schema}.{data_src}')
    # ==============================================================================
    # PROCESS
    # ==============================================================================
    logger.info('Start Process')
    
    logger.info('Start feature improtance')
    features = PersistManager.load_json_from_hdfs(sc, __CONFIG__['outputs']['feat_sel']['chosen'])
    features = features['chosen_feat']
    model = PersistManager.load_spark_obj(GBTClassificationModel(), __CONFIG__['outputs']['model'])
    logger.debug(features)
    logger.debug(model)
    
    pd_feat_import = ReportManager.get_feat_import(model, features)
    df_feat_import = spark.createDataFrame(pd_feat_import)
    PersistManager.save_as_csv_into_hdfs(df_feat_import, __CONFIG__['outputs']['feat_import'])
    
    logger.info('Start assessing performance')
    logger.info('Start metrics')
    evaluator = Evaluator(data=df, config=config)
    evaluator.split_data()
    evaluator.build_multi_eval()
    evaluator.build_bin_eval()
    
    metrics = []
    metrics.append(evaluator.get_metrics(split='train'))
    metrics.append(evaluator.get_metrics(split='test'))
    metrics.append(evaluator.get_metrics(split='val'))
    
    pd_metrics = ReportManager.merge_reports(metrics, on=['metric_name'])
    df_metrics = spark.createDataFrame(pd_metrics)
    PersistManager.save_as_csv_into_hdfs(df_metrics, __CONFIG__['outputs']['metrics'])
    
    logger.info('Start KS')
    df_ks_train = evaluator.get_KS(split='train')
    df_ks_test = evaluator.get_KS(split='test') 
    PersistManager.save_as_csv_into_hdfs(df_ks_train, __CONFIG__['outputs']['ks_train'])
    PersistManager.save_as_csv_into_hdfs(df_ks_test, __CONFIG__['outputs']['ks_test'])
    
    date_col = evaluator.config['spine']['date_col']
    df_val = evaluator.val
    row_dates = df_val.select(date_col).orderBy(f.col(date_col)).distinct().collect()
    val_dates = [row[0] for row in row_dates]
    for i, date in enumerate(val_dates):
        evaluator.val = df_val.filter(f.col(date_col) == date)
        df_ks_val = evaluator.get_KS(split='val')
        PersistManager.save_as_csv_into_hdfs(df_ks_val, __CONFIG__['outputs']['ks_val']+f'_{i}')
    
    logger.info('End process')
# ==================================================================================
# MAILING
# ==================================================================================   
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed assessing performance
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file]) 

except Exception as e:
    logger.error('------------------------------------------------\n')
    logger.error('EXECUTION ERROR\n')
    logger.error('------------------------------------------------\n')
    logger.error('ERROR: {}'.format(str(e)))
    end_time = time.time()
    
    subject = 'RUN: ' + app_name + ' - KO'
    message = f'''Dear colleagues,<br> Time elapsed in this attempt to assess performance
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])