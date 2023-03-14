import os
import time
import logging

def get_logger(loggername, save_path='.'):
    # create logger
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.INFO)
    save_path = save_path

    # create handler，for write log file
    log_path = os.path.join(save_path, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    strtime=time.strftime("%m_%d_%H_%M",time.localtime(time.time()))
    logname = os.path.join(log_path, f"{loggername}_{str(strtime)}.log")
    if not os.path.isfile(logname):
        f = open(logname,"w",encoding="utf-8")
        f.close()

    fh = logging.FileHandler(logname, encoding='utf-8')
    fh.setLevel(logging.INFO)

    # create handler, in order to output the log to the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # define the output format of the handler
    formatter = logging.Formatter('%(asctime)s | %(name)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
