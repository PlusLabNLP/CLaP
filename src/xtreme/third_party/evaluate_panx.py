from argparse import ArgumentParser
from seqeval.metrics import precision_score, recall_score, f1_score

def main():
    parser = ArgumentParser()
    parser.add_argument('-g', '--gold-file', type=str, required=True)
    parser.add_argument('-p', '--pred-file', type=str, required=True)
    parser.add_argument('--exclude-special', action='store_true')
    args = parser.parse_args()

    gold_labels, sent_labels = [], []
    with open(args.gold_file, 'r') as f:
        first_word = True
        for line in f:
            if args.exclude_special:
                if line.strip() == "":
                    gold_labels.append(sent_labels)
                    sent_labels = []
                    first_word = True
                elif first_word and (line.strip().split("\t")[0] == "=" or line.strip().split("\t")[0] == "+"):
                    continue
                else:
                    sent_labels.append(line.strip().split("\t")[1])
                    first_word = False
            else:
                if line.strip() == "":
                    gold_labels.append(sent_labels)
                    sent_labels = []
                else:
                    sent_labels.append(line.strip().split("\t")[1])
    if sent_labels != []:
        gold_labels.append(sent_labels)
        sent_labels = []
    
    pred_labels, sent_labels = [], []
    with open(args.pred_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                pred_labels.append(sent_labels)
                sent_labels = []
            else:
                sent_labels.append(line.strip())
    if sent_labels != []:
        pred_labels.append(sent_labels)
        sent_labels = []
    
    assert len(gold_labels) == len(pred_labels), (len(gold_labels), len(pred_labels))
    for i, (g, p) in enumerate(zip(gold_labels, pred_labels)):
        assert len(g) == len(p), (i, g, p)

    results = {
        "precision": precision_score(gold_labels, pred_labels),
        "recall": recall_score(gold_labels, pred_labels),
        "f1": f1_score(gold_labels, pred_labels)
    }
    print (results)
    
if __name__ == "__main__":
    main()
