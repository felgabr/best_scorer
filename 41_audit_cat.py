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
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassificationModel, GBTClassifier
from pyspark.ml.classification import RandomForestClassificationModel, RandomForestClassifier
from pyspark.ml.feature import OneHotEncoderEstimator, OneHotEncoderModel, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
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
parser.add_argument('-s', '--split', type=str, default='train')
args = parser.parse_args()
# GLOBAL
abs_path = os.getcwd()
user = os.environ['USER']
# Load config file
conf_filepath = os.path.join(abs_path, args.config)
__CONFIG__ = json.load(open(conf_filepath))
# General
project = __CONFIG__['project']
app_name = f'{project}_auditcat'
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
from audit import Auditor
from datawrangling import Cleaner
from supplementary import DateAux, LakeAux, MergerDF, ReportManager, PersistManager
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
report = __CONFIG__['rep_auditcat']
report_outpath = f'{report}_{args.split}.xlsx'
# ML config
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
    logger.info(f'Inital date: {ini_date}')
    logger.info(f'End date: {end_date}')    
    
    master_src = LakeAux.get_most_recent_table(spark, schema, master_src)
    df = spark.sql(f"""SELECT
                        *
                    FROM
                        {schema}.{master_src}
                    WHERE
                        ind_split = '{args.split}'
                    """)
    
    merger = MergerDF(spark, config, schema, df, data_src, 'idnt_mensual',
                       'date_part_mensual', ini_date, end_date)
    df = merger()    
    
    features = ReportManager.get_original_feat_names(sc, path_feat)
    features.append(config['spine']['date_col'])
    
    df = df.select(*features)   
    # ==============================================================================
    # PROCESS
    # ==============================================================================
    logger.info('Start Process')
    
    cleaner = Cleaner(df, config)
    
    cleaner.repart()
    cleaner.rmv_whitespace()
    
    df = cleaner.data
    
    dfs_list = []
    dates = df.select(f.col(config['spine']['date_col'])).orderBy(f.col(config['spine']['date_col'])).distinct().collect()
    for date in dates:
        df_tmp = df.filter(f.col(config['spine']['date_col']) == date[0])

        dfs_list.append(Auditor.audit_varcat(df_tmp, date[0], not_eval=list(config['spine'].values())))

    pd_merged = pd.concat(dfs_list, axis=0, ignore_index=True)        
    pd_merged.to_excel(report_outpath, index=False)

    logger.info(PersistManager.copy_from_local_to_hdfs(report_outpath, project))
    
    logger.info('End process')
# ==================================================================================
# MAILING
# ==================================================================================    
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed auditing numerical variables of {args.split}
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file, report_outpath])

except Exception as e:
    logger.error('------------------------------------------------\n')
    logger.error('EXECUTION ERROR\n')
    logger.error('------------------------------------------------\n')
    logger.error('ERROR: {}'.format(str(e)))
    end_time = time.time()
    
    subject = 'RUN: ' + app_name + ' - KO'
    message = f'''Dear colleagues,<br> Time elapsed in this attempt to audit numerical variables of {args.split}
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])