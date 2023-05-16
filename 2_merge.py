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
app_name = f'{project}_merge'
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
from supplementary import AuxHandleData, DateAux, Merger, PersistManager
from utils import build_spark_session, send_email

from funciones_analista import create_spark_session, f_envia_mail
# ==================================================================================
# PARAMETERS pt2
# ==================================================================================
# Date Parameters
exec_date = (datetime.datetime.today()).strftime('%Y%m%d_%H%M%S')
# Tables
data_src = [v for k, v in __CONFIG__['table_outputs'].items() if k.startswith('clean')]
master_src = __CONFIG__['table_outputs']['spine']
# Mailing
dest = __CONFIG__['from']
recp = __CONFIG__['to']
recp_cp = __CONFIG__['cc']
# Output
table_name = __CONFIG__['table_outputs'][f'dataset']
main_fullname = f'{table_name}_{exec_date}'
# ==================================================================================
# MAIN
# ==================================================================================
try:
    logger.info('Running')
    logger.info(f'Hostname: {socket.gethostname()}')

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
    # PROCESS
    # ==============================================================================
    logger.info('Start process')
    
    merger = Merger(spark, __CONFIG__['conf'], schema, master_src, data_src)
    df, interm_tbs = merger()    
    
    logger.info('Start save data')
    logger.info(PersistManager.save_df_into_lake(spark, df, main_fullname, schema,
                                                 particiona_cols=__CONFIG__['new_part_cols']))
    
    logger.info('Start drop intermediary tables')
    df = spark.read.table(f'{schema}.{main_fullname}')
    not_empty = AuxHandleData().check_isnot_empty(df)
    if not_empty:
        for tb in interm_tbs:
            spark.sql(f'drop table if exists {tb}')
    
    logger.info('End save data')
# ==================================================================================
# MAILING
# ==================================================================================
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed merging cleaned tables
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
    message = f'''Dear colleagues,<br> Time elapsed in this attempt to merge cleaned tables
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])