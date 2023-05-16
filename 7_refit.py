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
app_name = f'{project}_refit'
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
from supplementary import DateAux, MergerRaw, ReportManager, PersistManager
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
data_src = [v for k, v in __CONFIG__['table_sources'].items() if k.startswith('clean')]
master_src = __CONFIG__['table_outputs']['spine']
# Mailing
dest = __CONFIG__['from']
recp = __CONFIG__['to']
recp_cp = __CONFIG__['cc']
# Output
idx_path = __CONFIG__['outputs']['idx']
idx_fit_path = __CONFIG__['outputs']['idx_fit']
ohe_path = __CONFIG__['outputs']['ohe']
ohe_fit_path = __CONFIG__['outputs']['ohe_fit']
# ML config
feat_sel_mod = __CONFIG__['conf']['feat_mode']
config = __CONFIG__['conf']
path_feat = __CONFIG__['outputs']['feat_sel']['chosen']
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

    merger = MergerRaw(spark, config, schema, master_src, data_src,
                       __CONFIG__['mode'], 'cliident_client_mensual',
                       'data_date_part_client_mensual', ini_date, end_date)
    df = merger()
    
    features = ReportManager.get_original_feat_names(sc, path_feat)
    spine = [v for k, v in config['spine'].items() if k.startswith('ident') or k.startswith('date')]
    features = spine + features
    logger.debug(features)
    
    df = df.select(*features)    
    # ==============================================================================
    # PROCESS
    # ==============================================================================
    logger.info('Start Process')
    
    logger.info('Start Cleaning')
    cleaner = Cleaner(df, config)

    cleaner.repart()
    cleaner.rmv_whitespace()
    cleaner.to_upper()

    logger.info('Start binary encoding')
    cntdist = cleaner.get_cnt_dist()
    cleaner.cnt_dist = cntdist
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
    preprocess.transform_data(obj='idx')

    logger.info('Start one hot encoding')
    preprocess.fit_ohe()
    preprocess.transform_data(obj='ohe', split='train')

    logger.info('Start Save Objects')
    PersistManager.save_spark_obj(preprocess.indxr_fitted, idx_fit_path)
    PersistManager.save_spark_obj(preprocess.ohe, ohe_path)
    PersistManager.save_spark_obj(preprocess.ohe_fitted, ohe_fit_path)    

    logger.info('End process')
# ==================================================================================
# MAILING
# ==================================================================================    
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed refitting pipeline objects
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
    message = f'''Dear colleagues,<br> Time elapsed in this attempt to refit pipeline objects
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])