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
from datawrangling import Cleaner
from mlfeatures import ModelProcessor, Scorer
from supplementary import DateAux, MergerRaw, ReportManager, PersistManager
from utils import build_spark_session, send_email
# ==================================================================================
# PARAMETERS pt2
# ==================================================================================
# Date Parameters
dt_format = '%Y-%m-%d'
ini_date = DateAux.get_first_day_month(ref_date=datetime.datetime.today().strftime(dt_format),
                                       rfd_format=dt_format, m_offset=-3)
end_date = DateAux.get_last_day_month(ref_date=ini_date, rfd_format=dt_format)
exec_date = (datetime.datetime.today()).strftime('%Y%m%d_%H%M%S')
# Tables
data_src = [v for k, v in __CONFIG__['table_sources'].items() if k.startswith('clean')]
master_src = __CONFIG__['table_sources']['spine_prod']
# Mailing
dest = __CONFIG__['from']
recp = __CONFIG__['to']
recp_cp = __CONFIG__['cc']
# Source
idx_path = __CONFIG__['outputs']['idx']
idx_fit_path = __CONFIG__['outputs']['idx_fit']
ohe_path = __CONFIG__['outputs']['ohe']
ohe_fit_path = __CONFIG__['outputs']['ohe_fit']
model = __CONFIG__['outputs']['model']
# Output
potentials = __CONFIG__['table_outputs']['spine_prod']
potentials_path = f'{potentials}_{exec_date}'
table_score_path = __CONFIG__['table_outputs']['prod']
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
    
    df = spark.sql(f"""SELECT
                        *
                    FROM
                        {master_src}
                    WHERE
                        date_part between '{ini_date}' AND '{end_date}'
                    """)
    
    logger.info(PersistManager.save_df_into_lake(spark, df, potentials_path, schema,
                                                 particiona_cols=__CONFIG__['new_part_cols']))
    
    merger = MergerRaw(spark, config, schema, potentials, data_src,
                       __CONFIG__['mode'], 'idnt_mensual',
                       'date_part_mensual', ini_date, end_date)
    df = merger()    
    
    features = ReportManager.get_original_feat_names(sc, path_feat)
    spine = [v for k, v in config['spine'].items() if k.startswith('ident') or k.startswith('date')]
    features = spine + features
    
    df = df.select(*features)
    # ==============================================================================
    # LOAD OBJECTS
    # ==============================================================================
    logger.info('Start load objects')
    
    indxr_fitted = PersistManager.load_spark_obj(PipelineModel(stages=[]), __CONFIG__['outputs']['idx_fit'])
    
    ohe = PersistManager.load_spark_obj(OneHotEncoderEstimator, __CONFIG__['outputs']['ohe'])
    ohe_fitted = PersistManager.load_spark_obj(OneHotEncoderModel(), __CONFIG__['outputs']['ohe_fit'])
    
    assembler = PersistManager.load_spark_obj(VectorAssembler, __CONFIG__['outputs']['assembler'])
    model = PersistManager.load_spark_obj(GBTClassificationModel(), __CONFIG__['outputs']['model'])
    
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
    preprocess = ModelProcessor(data=df, config=config,
                                indxr_fitted=indxr_fitted,
                                ohe=ohe, ohe_fitted=ohe_fitted,
                                assembler=assembler, model=model) 
    
    logger.info('Start Transform data')
    preprocess.transform_data(obj='idx')
    preprocess.transform_data(obj='ohe')
    preprocess.get_dummies()
    preprocess.transform_data(obj='assembler')
    preprocess.transform_data(obj='model')
 
    df = preprocess.data
    
    logger.info('Start Scoring')
    scorer = Scorer(data=df, config=config)
    
    scorer.score_proba()
    scorer.score_logit(score_name='score_alt')
    scorer.get_percentiles()
    scorer.create_ts_col()

    df = scorer.data
    
    try:
        logger.info(PersistManager.append_partition(spark, df, table_score_path, schema,))
    except:
        logger.info(PersistManager.save_df_into_lake(spark, df, table_score_path, schema,
                                                     particiona_cols=__CONFIG__['new_part_cols']))
        
    spark.sql(f'drop table if exists {potentials_path}')
    
    logger.info('End process')
# ==================================================================================
# MAILING
# ==================================================================================    
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed predicting {ini_date}
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
    message = f'''Dear colleagues,<br> Time elapsed in this attempt to predict {ini_date}
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])