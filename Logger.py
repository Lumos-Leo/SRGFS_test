import sys
import datetime
import os
import logging


class Logger():
    def __init__(self, args, flush=True):
        if not args.test_only and flush:
            if args.model == 'SESR':
                experiment_name = "{}-x{}-m{}-c{}-p{}-s{}-a{}-{}".format(args.model, args.scale, args.n_resblocks, args.n_feats, args.collapse_rate*args.n_feats, args.sparsity_target, args.alpha, self.cur_timestamp_str())
            else:
                experiment_name = "{}-x{}-m{}-c{}-p{}-s{}-a{}-{}".format(args.model, args.scale, args.n_resblocks, args.n_feats, args.collapse_rate*args.n_feats, args.sparsity_target, args.alpha, self.cur_timestamp_str())
            self.experiment_path = os.path.join(args.log_dir, experiment_name)
            if not os.path.exists(self.experiment_path):
                os.makedirs(self.experiment_path)
            self.experiment_model_path = os.path.join(self.experiment_path, 'models')
            if not os.path.exists(self.experiment_model_path):
                os.makedirs(self.experiment_model_path)

            self.log_name = os.path.join(self.experiment_path, "log.log")
        
            # init logger
            self.logger = logging.getLogger()
            self.logger.setLevel(level=logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            streamHandler = logging.StreamHandler()
            streamHandler.setLevel(logging.DEBUG)
            streamHandler.setFormatter(formatter)
            self.logger.addHandler(streamHandler)

            fileHandler = logging.FileHandler(self.log_name)
            fileHandler.setLevel(logging.INFO)
            fileHandler.setFormatter(formatter)
            self.logger.addHandler(fileHandler)
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel(level=logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            streamHandler = logging.StreamHandler()
            streamHandler.setLevel(logging.DEBUG)
            streamHandler.setFormatter(formatter)
            self.logger.addHandler(streamHandler)

    def cur_timestamp_str(self):
        now = datetime.datetime.now()
        year = str(now.year)
        month = str(now.month).zfill(2)
        day = str(now.day).zfill(2)
        hour = str(now.hour).zfill(2)
        minute = str(now.minute).zfill(2)

        content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
        return content


    