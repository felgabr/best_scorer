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
app_name = f'{project}_train'
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
from mlfeatures import Fitter
from supplementary import Criterion, DateAux, LakeAux, PersistManager
from utils import build_spark_session, send_email
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
model_path = __CONFIG__['outputs']['model']
assembler_path = __CONFIG__['outputs']['assembler']
output_table = __CONFIG__['table_outputs']['pred']
output_fullname = f'{output_table}_{exec_date}'
# ML config
feat_sel_mod = __CONFIG__['conf']['feat_mode']
config = __CONFIG__['conf']
crit = __CONFIG__['conf']['criterion']
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
    
    logger.info('Start Choosing features')
    if feat_sel_mod == 'infogain':
        stat_col = 'importance'
        df_feat_selec = PersistManager.read_csv_from_hdfs(spark,
                                                          __CONFIG__['outputs']['feat_sel']['infogain'])
    elif feat_sel_mod == 'chisqrd':
        df_feat_selec = PersistManager.read_csv_from_hdfs(spark,
                                                          __CONFIG__['outputs']['feat_sel']['chisqrd'])
    elif feat_sel_mod == 'cramer':
        stat_col = 'cramer_v'
        df_feat_selec = PersistManager.read_csv_from_hdfs(spark,
                                                          __CONFIG__['outputs']['feat_sel']['cramer'])
    elif feat_sel_mod == 'mtinfo':
        stat_col = 'mutual_information'
        df_feat_selec = PersistManager.read_csv_from_hdfs(spark,
                                                          __CONFIG__['outputs']['feat_sel']['mtinfo'])

    pd_feat_selec = df_feat_selec.toPandas()
    if feat_sel_mod == 'chisqrd':
        features = list(pd_feat_selec['feature'])
    else:
        criterion = Criterion(pd_feat_selec)
        
    if crit == 'top':
        features = criterion.top_criteron(top=config['top'], imp_col=stat_col)
    elif crit == 'percentile':
        features = criterion.percentile_criterion(percentile=config['percentile'], imp_col=stat_col)
    elif crit == 'threshold':
        features = criterion.threshold_criterion(threshold=config['threshold'], imp_col=stat_col)
    elif crit == 'perc':
        features = criterion.perc_criterion(perc=config['perc'], imp_col=stat_col)

    logger.info('Start save input features')
    dict_feat = {'chosen_feat': features}
    PersistManager.save_as_json(spark, sc, dict_feat, __CONFIG__['outputs']['feat_sel']['chosen'])
    
    logger.info('Start Modeling')    
    gbtc = GBTClassifier(labelCol=config['spine']['target_col'],
                         featuresCol='features',
                         seed=2012,
                         maxDepth=5,
                         maxBins=32,
                         maxIter=50,
                         stepSize=0.1)
    logger.debug(gbtc)
    
    fitter = Fitter(data=df, config=config, feat_names=features, model=gbtc)
    fitter.subset_data()
    fitter.split_data()
    fitter.train_transform_model()
    
    logger.info('Start Save Objects')
    PersistManager.save_spark_obj(fitter.model, model_path)
    PersistManager.save_spark_obj(fitter.assembler, assembler_path)
    
    logger.info('Start Predicting')
    fitter.transform_data(obj='assembler', split='test')
    fitter.transform_data(obj='assembler', split='val')
    fitter.transform_data(obj='model', split='test')
    fitter.transform_data(obj='model', split='val')
    
    fitter.feat_names = ['probability', 'rawprediction', 'prediction'] 
    
    logger.info('Start save data')
    fitter.train = fitter.subset_data(fitter.train)
    fitter.test = fitter.subset_data(fitter.test)
    fitter.val = fitter.subset_data(fitter.val)
    df = fitter.merge_train_test_val()
    
    logger.info(PersistManager.save_df_into_lake(spark, df, output_fullname, schema,
                                                 particiona_cols=__CONFIG__['new_part_cols']))
    
    logger.info('End process')
# ==================================================================================
# MAILING
# ==================================================================================    
    logger.info('Start send email')
    end_time = time.time()
    subject = 'RUN: ' + app_name + ' - OK'
    message = f'''Dear colleagues,<br> Time elapsed training model and predicting data
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
    message = f'''Dear collesgues,<br> Time elapsed in this attempt to train the model and predict the data
                  was {DateAux.time_elapsed(end_time, start_time)} hours.
                  <br> <br> Best Regards.''' 
    send_email(sender=dest, to=recp, cc=recp_cp, message=message, subject=subject,
                 attch=[log_file, stdlog_file])