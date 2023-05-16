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
app_name = f'{project}_prod'
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
from supplementary import DateAux, PersistManager
from utils import build_spark_session, send_email
# ==================================================================================
# PARAMETERS pt2
# ==================================================================================
# Tables
prod_src = __CONFIG__['table_outputs']['prod']
target_src = __CONFIG__['table_sources']['spine_prod']
# Mailing
dest = __CONFIG__['from']
recp = __CONFIG__['to']
recp_cp = __CONFIG__['cc']
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
    
    df = spark.sql(f"""SELECT
                        *
                    FROM
                        {schema}.{prod_src}
                    WHERE
                        {config['spine']['date_col']} = (SELECT max({config['spine']['date_col']}) FROM {schema}.{prod_src})
                        AND exec_time = (SELECT max(exec_time) FROM {schema}.{prod_src})
                    """)
    
    df_target = spark.read.table(target_src)
    
    df = df.join(df_target, [config['spine']['ident_col'], config['spine']['date_col']])    
    # ==============================================================================
    # PROCESS
    # ==============================================================================
    logger.info('Start assessing performance')
    logger.info('Start metrics')
    evaluator = Evaluator(data=df, config=config)
    evaluator.build_multi_eval()
    evaluator.build_bin_eval()
    
    pd_metrics = evaluator.get_metrics()
    df_metrics = spark.createDataFrame(pd_metrics)
    PersistManager.save_as_csv_into_hdfs(df_metrics, __CONFIG__['prod_path'])
    
    logger.info('Start KS')
    df_ks = evaluator.get_KS()
    PersistManager.save_as_csv_into_hdfs(df_ks, __CONFIG__['prod_path'])
    
    logger.info('End process')
# ==================================================================================
# MAILING
# ==================================================================================    
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed evaluating last prediction
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
    message = f'''Dear colleagues,<br> Time elapsed in this attempt to evaluate last prediction
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])