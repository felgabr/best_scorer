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
app_name = f'{project}_clean'
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
from datawrangling import Cleaner
from mlfeatures import ModelProcessor
from supplementary import DateAux, LakeAux, PersistManager
from utils import build_spark_session, send_email
# ==================================================================================
# PARAMETERS pt2
# ==================================================================================
# Date Parameters
m_offset = __CONFIG__['qty_months']
ini_date, ini_date_format = __CONFIG__['ini_date']
end_date = DateAux.get_last_day_month(ref_date=ini_date, rfd_format=ini_date_format, m_offset=m_offset)
exec_date = (datetime.datetime.today()).strftime('%Y%m%d_%H%M%S')
# Tables
index_table = 0
data_src = __CONFIG__['table_sources'][f'clean_{index_table}']
master_src = __CONFIG__['table_outputs']['spine']
# Mailing
dest = __CONFIG__['from']
recp = __CONFIG__['to']
recp_cp = __CONFIG__['cc']
# Output
table_name = __CONFIG__['table_outputs'][f'clean_{index_table}']
cleaned_fullname = f'{table_name}_{exec_date}'
# ML config
config = __CONFIG__['conf']
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
    # LOAD DATA
    # ==============================================================================
    logger.info('Start load data')
    logger.info(f'Data source: {data_src}')
    logger.info(f'Inital date: {ini_date}')
    logger.info(f'End date: {end_date}')

    master_src = LakeAux.get_most_recent_table(spark, schema, master_src)
    logger.info(f'Master: {master_src}')
    df = spark.read.table(f'{schema}.{master_src}')
    
    
    df_info = spark.sql(f"""SELECT
                                *
                            FROM
                                {data_src}
                            WHERE
                                date_part between '{ini_date}' AND 
                                                                            '{end_date}'
                            """)
    df_info = df_info.drop(*__CONFIG__['feat_drop'])
    
    df = df.join(df_info, ['ident', 'data_date'], how='left')    
    # ==============================================================================
    # PROCESS
    # ==============================================================================
    logger.info('Start Cleaning')
    
    cleaner = Cleaner(df, config)
    
    cleaner.repart()
    cleaner.rmv_whitespace()
    cleaner.to_upper()
    
    logger.info('Start drop nulls')
    dnull = cleaner.get_perc_null()
    cleaner.perc_null = dnull
    cleaner.drop_null()
    
    logger.info('Start drop quasi constant')
    cntdist = cleaner.get_cnt_dist()
    cleaner.cnt_dist = cntdist
    dpmode = cleaner.get_mode_perc()
    cleaner.pmode = dpmode
    cleaner.drop_mode_threshold()
    
    logger.info('Start binary encoding')
    bin_cols, bin_cols_enc = cleaner.idntf_bin_vars()
    cleaner.bin_vars = bin_cols
    cleaner.bin_enc = bin_cols_enc
    if len(cleaner.bin_vars) > 0 or len(cleaner.bin_enc) > 0:
        cleaner.enc_bin_vars()
    
    logger.info('Start removing special characters')
    cleaner.reg_replace(' ', '_')
    cleaner.reg_replace('[^a-zA-Z0-9_]', '')
    
    logger.info('Start fill nulls')
    cleaner.fill_null(0, cleaner._get_num_cols())
    cleaner.fill_null('SIN_INFO', cleaner._get_str_cols()+cleaner.config['bin_input'])
    
    df = cleaner.data
    
    logger.info('Start preprocess')
    preprocess = ModelProcessor(data=df, config=config)
    preprocess.split_data()
    
    logger.info('Start StringIndexer')
    preprocess.fit_str_indexer()
    preprocess.transform_data(obj='idx', split='train')
    preprocess.transform_data(obj='idx', split='test')
    preprocess.transform_data(obj='idx', split='val')
    
    logger.info('Start one hot encoding')
    preprocess.fit_ohe()
    preprocess.transform_data(obj='ohe', split='train')
    preprocess.transform_data(obj='ohe', split='test')
    preprocess.transform_data(obj='ohe', split='val')
    preprocess.get_dummies(split='train')
    preprocess.get_dummies(split='test')
    preprocess.get_dummies(split='val')
    
    logger.info('Start save data')
    df = preprocess.merge_train_test_val()
    logger.info(PersistManager.save_df_into_lake(spark, df, cleaned_fullname, schema,
                                                 particiona_cols=__CONFIG__['new_part_cols']))
    
    logger.info('End process')
# ==================================================================================
# MAILING
# ==================================================================================
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed cleaning {data_src}
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
    message = f'''Dear colleagues,<br> Time elapsed in this attempt to clean {data_src}
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])