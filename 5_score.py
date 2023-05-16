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
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
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
app_name = f'{project}_score'
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
from mlfeatures import Scorer
from supplementary import AuxHandleData, DateAux, LakeAux, PersistManager
from utils import build_spark_session, send_email
# ==================================================================================
# PARAMETERS pt2
# ==================================================================================
# Date Parameters
exec_date = (datetime.datetime.today()).strftime('%Y%m%d_%H%M%S')
# Tables
data_src = __CONFIG__['table_outputs'][f'pred']
# Mailing
dest = __CONFIG__['from']
recp = __CONFIG__['to']
recp_cp = __CONFIG__['cc']
# Output
output_table = __CONFIG__['table_outputs']['score']
output_fullname = f'{output_table}_{exec_date}'
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
    
    scorer = Scorer(data=df, config=config)
    scorer.split_data()
    
    logger.info('Start Score Proba 1')
    scorer.score_proba(split='train')
    scorer.score_proba(split='test')
    scorer.score_proba(split='val')

    logger.info('Start Score logit custom')
    scorer.score_logit(split='train', score_name='score_alt')
    scorer.score_logit(split='test', score_name='score_alt')
    scorer.score_logit(split='val', score_name='score_alt')
    
    logger.info('Start get percentiles')
    scorer.get_percentiles(split='train')
    scorer.get_percentiles(split='test')
    scorer.get_percentiles(split='val')
    
    logger.info('Start save data')
    df = scorer.merge_train_test_val()
    
    logger.info(PersistManager.save_df_into_lake(spark, df, output_fullname, schema,
                                                 particiona_cols=__CONFIG__['new_part_cols']))
    
    logger.info('Start drop intermediary tables')
    df = spark.read.table(f'{schema}.{output_fullname}')
    not_empty = AuxHandleData().check_isnot_empty(df)
    if not_empty:
        spark.sql(f'drop table if exists {schema}.{data_src}')

    logger.info('End process')
# ==================================================================================
# MAILING
# ==================================================================================   
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed scoring
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
    message = f'''Dear colleagues,<br> Time elapsed in this attempt to score data
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])