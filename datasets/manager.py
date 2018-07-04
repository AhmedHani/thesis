import sys
sys.path.append('../')

import argparse
from datasets.quora import QuoraQuestionsPairDataset
import vars

parser = argparse.ArgumentParser(description='This script is responsible for downloading '
                                             'and preparing the needed datasets')

parser.add_argument('--datasets',
                    default='quora,mscoco,ppdb',
                    type=str,
                    help='dataset1,dataset2,dataset3')

args = parser.parse_args()
datasets = args.datasets

chosen = str(datasets).split(',')

for choice in chosen:
    if choice == 'quora':
        q = QuoraQuestionsPairDataset(download_link=vars.QUORA_QUESTIONS_PAIRS_DOWNLOAD_LINK)

print('Success!')
