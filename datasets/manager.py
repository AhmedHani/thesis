import sys

sys.path.append('../')

import argparse
from datasets.quora import QuoraQuestionsPairDataset
from datasets.snli import SNLIDataset
from datasets.ppdb import PPDBDataset
import vars

parser = argparse.ArgumentParser(description='This script is responsible for downloading '
                                             'and preparing the needed datasets')

parser.add_argument('--datasets',
                    default='quora,mscoco,ppdb,snli',
                    type=str,
                    help='dataset1,dataset2,dataset3')

args = parser.parse_args()
datasets = args.datasets

chosen = str(datasets).split(',')

for choice in chosen:
    if choice == 'quora':
        q = QuoraQuestionsPairDataset(download_link=vars.QUORA_QUESTIONS_PAIRS_DOWNLOAD_LINK)
    elif choice == 'snli':
        snli = SNLIDataset(download_link=vars.SNLI_DOWNLOAD_LINK)
    elif choice == 'ppdb':
        print('sdsd')
        ppdb = PPDBDataset(download_link=vars.PPDB_DOWNLOAD_LINK)

print('Success!')
