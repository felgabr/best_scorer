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
import pyspark.sql.types as t
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
#user = os.environ['USER']
# Load config file
conf_filepath = os.path.join(abs_path, args.config)
__CONFIG__ = json.load(open(conf_filepath))
# General
project = __CONFIG__['project']
app_name = f'{project}_spine'
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
from supplementary import DateAux, PersistManager, ReportManager
from mlfeatures import SplitTrainTestVal
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
data_src = __CONFIG__['table_sources']['spine']
# Mailing
dest = __CONFIG__['from']
recp = __CONFIG__['to']
recp_cp = __CONFIG__['cc']
# Output Paths
prior_path = __CONFIG__['outputs']['prior']
master_name = __CONFIG__['table_outputs']['spine']
master_fullname = f'{master_name}_{exec_date}'
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
                
    df = spark.sql(f"""SELECT
                        *
                    FROM
                        {data_src}
                    WHERE
                        date_part between '{ini_date}' AND '{end_date}'
                    """)
    # Ensure Target is int
    df = df.withColumn('target', f.col('target').cast(t.IntegerType()))
    # ==============================================================================
    # PROCESS
    # ==============================================================================
    logger.info('Start Process')
    
    splitter = SplitTrainTestVal(df, __CONFIG__['conf'])

    splitter.create_id_col()
    
    logger.info('Start Split Train / Validation')
    splitter.split_train_val()
    logger.info('Start Split Train with sampling and Test')
    dict_prior_orig, dict_prior = splitter.split_train_test_bymonth(spark)
    splitter.repart()
    # Saving Prior
    pd_prior = ReportManager.gen_prior_report(dict_prior_orig, dict_prior)
    df_prior = spark.createDataFrame(pd_prior)
    PersistManager.save_as_csv_into_hdfs(df_prior, prior_path)
    logger.info('Prior exported successfully')
    
    splitter.create_ind_split()
    df = splitter.get_joined_df()
    
    logger.info('Start save data')
    logger.info(PersistManager.save_df_into_lake(spark, df, master_fullname, schema,
                                                 particiona_cols=__CONFIG__['new_part_cols']))

    logger.info('End process')
# ==================================================================================
# MAILING
# ==================================================================================
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed creating spine
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
    message = f'''Dear colleagues,<br> Time elapsed in this attempt to create spine
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])