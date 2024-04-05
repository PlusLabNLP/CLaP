import os, logging, json
from argparse import ArgumentParser
from utils import TRAINER_MAP, load_config, load_all_data, set_seed, set_gpu, set_logger, load_trans_data
from scorer import compute_scores, print_scores
import ipdb

logger = logging.getLogger(__name__)

def main():
    # configuration
    parser = ArgumentParser()
    parser.add_argument('-c', '--config')
    args = parser.parse_args()
    config = load_config(args.config)
    config.output_dir = config.output_dir + "/s%d" % config.seed

    set_seed(config.seed)
    set_gpu(config.gpu_device)
    config = set_logger(config)

    trainer_class = TRAINER_MAP[(config.model_type, config.task)]

    # load data
    train_data, dev_data, test_data, type_set = load_all_data(config, trainer_class.add_extra_info_fn)

    # load translated data
    trans_data = load_trans_data(config)
    
    # train
    trainer = trainer_class(config, type_set)
    best_model_checkpoint = trainer.train(train_data, dev_data, trans_data=trans_data)
    logger.info("Training was done!")
    
    # test
    logger.info("Loading best model for evaluation.")
    trainer.load_model(checkpoint=best_model_checkpoint)
    predictions = trainer.predict(test_data)
    scores = compute_scores(predictions, test_data, config.task)
    print("Test")
    print_scores(scores)
        
if __name__ == "__main__":
    main()
    
    
