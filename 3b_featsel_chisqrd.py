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
app_name = f'{project}_chisqrd'
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
from mlfeatures import FeatureSelector
from supplementary import DateAux, LakeAux, PersistManager
from utils import build_spark_session, send_email
from funciones_analista import create_spark_session, f_envia_mail
# ==================================================================================
# PARAMETERS pt2
# ==================================================================================
# Date Parameters
exec_date = (datetime.datetime.today()).strftime('%Y%m%d_%H%M%S')
# Tables
data_src = __CONFIG__['table_outputs'][f'dataset']
# Mailing
dest = __CONFIG__['from']
recp = __CONFIG__['to']
recp_cp = __CONFIG__['cc']
# Output
featsel_path = __CONFIG__['outputs']['feat_sel']['chisqrd']
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
    df = df.filter(f.col(config['spine']['ind_split_col']) == 'train')
    # ==============================================================================
    # PROCESS
    # ==============================================================================
    logger.info('Start Process')

    selector = FeatureSelector(data=df, config=config)
    
    cntdist = selector.get_cnt_dist()
    selector.cnt_dist = cntdist
    bin_cols, _ = selector.idntf_bin_vars()  
    excpt_list = bin_cols + config['spine'].values()
    
    logger.info('Start ChiSquared Selection')
    selector.split_data()
    pd_results = selector.chisqr_selctr(excpt_list=excpt_list,
                                        buckets=5,
                                        error=0,
                                        handler='error',
                                        numTopFeatures=100,
                                        featuresCol='features',
                                        outputCol='selc_feat',
                                        labelCol=config['spine']['target_col'])
    
    logger.info('Start exporting')
    df_results = spark.createDataFrame(pd_results)
    PersistManager.save_as_csv_into_hdfs(df_results, featsel_path)
    
    logger.info('End process')
# ==================================================================================
# MAILING
# ==================================================================================
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed doing feature selecting using ChiSquared
                  Selector was {DateAux.time_elapsed(end_time, start_time)} hours.
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
    message = f'''Dear colleagues,<br> Time elapsed in this attempt to do feature selection using ChiSquared
                  Selector was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])