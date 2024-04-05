import os, logging, json, pprint
from argparse import ArgumentParser
from utils import TRAINER_MAP, load_config, load_data, set_gpu, convert_ED_to_EAE, combine_ED_and_EAE_to_E2E
from scorer import compute_scores, print_scores
from pathlib import Path
import ipdb

logger = logging.getLogger(__name__)

def main():
    # configuration
    parser = ArgumentParser()
    parser.add_argument('--test_config', required=True, help="For cross-lingual testing, please provide corresponding testing config for testing.")
    parser.add_argument('--eae_model', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save-preds', action="store_true")
    args = parser.parse_args()

    set_gpu(args.gpu)
    test_config = load_config(args.test_config)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]',
                        handlers=[logging.FileHandler(os.path.join(Path(args.eae_model).parent, "eval_%s.log" % test_config.lang)), logging.StreamHandler()])

    # load model
    assert args.eae_model
    eae_config = load_config(os.path.join(Path(args.eae_model).parent, "config.json"))
    eae_config.gpu_device = args.gpu
    logger.info(f"\n{pprint.pformat(vars(eae_config), indent=4)}")
    eae_trainer_class = TRAINER_MAP[(eae_config.model_type, "EAE")]

    # load data
    eval_data, _ = load_data("EAE", test_config.test_file, eae_trainer_class.add_extra_info_fn, test_config)
    
    eae_trainer = eae_trainer_class(eae_config)
    eae_trainer.load_model(checkpoint=args.eae_model)
    
    # predict
    predictions = eae_trainer.predict(eval_data)
    scores = compute_scores(predictions, eval_data, "EAE")
    print("Evaluate")
    print_scores(scores)
    logger.info(pprint.pformat({"eval_scores": scores}))

    if args.save_preds is not None:
        filename_str = "test_%s.pred.json"
        print ("Saving file to: ", os.path.join(Path(args.eae_model).parent, filename_str % test_config.lang))
        with open(os.path.join(Path(args.eae_model).parent, filename_str % test_config.lang), 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
    
    
