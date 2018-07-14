import vars
import os
import shutil
import codecs


class SNLIDataset(object):

    def __init__(self, download_link=vars.SNLI_DOWNLOAD_LINK):
        if not self.__zip_existed():
            self.__create_dir()
            self.__download_and_extract(download_link)

        if not self.__moddata_exist():
            self.__toformat()

    @staticmethod
    def __download_and_extract(download_link):
        import zipfile, urllib.request

        url = download_link
        file_name = vars.SNLI_ZIP

        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

            with zipfile.ZipFile(file_name) as zf:
                zf.extractall()

    @staticmethod
    def __zip_existed():
        if os.path.exists(vars.SNLI_ZIP):
            return True

        return False

    @staticmethod
    def __create_dir():
        if os.path.exists(vars.SNLI_DIR):
            shutil.rmtree(vars.SNLI_DIR)

        os.mkdir(vars.SNLI_DIR)

    @staticmethod
    def __moddata_exist():
        if os.path.exists(vars.SNLI_TRAIN_MOD) \
                or os.path.exists(vars.SNLI_DEV_MOD) \
                or os.path.exists(vars.SNLI_TEST_MOD):
            return True

        return False

    @staticmethod
    def __toformat():
        import json

        with codecs.open(os.path.join(vars.SNLI_TRAIN_RAW), mode='r', encoding='utf-8') as reader:
            with codecs.open(vars.SNLI_TRAIN_MOD, 'w', encoding='utf-8') as writer:
                train_datalist = []

                for line in reader:
                    dic_line = json.loads(line)
                    label = dic_line['gold_label']
                    sentence1 = dic_line['sentence1']
                    sentence2 = dic_line['sentence2']

                    if label == 'neutral' or label == 'entailment':
                        writer.write(sentence1 + '\t' + sentence2 + '\n')

                    train_datalist.append([sentence1, sentence2, label])

        with codecs.open(os.path.join(vars.SNLI_DEV_RAW), mode='r', encoding='utf-8') as reader:
            with codecs.open(vars.SNLI_DEV_MOD, 'w', encoding='utf-8') as writer:
                dev_datalist = []

                for line in reader:
                    dic_line = json.loads(line)
                    label = dic_line['gold_label']
                    sentence1 = dic_line['sentence1']
                    sentence2 = dic_line['sentence2']

                    if label == 'neutral' or label == 'entailment':
                        writer.write(sentence1 + '\t' + sentence2 + '\n')

                    dev_datalist.append([sentence1, sentence2, label])

        with codecs.open(os.path.join(vars.SNLI_TEST_RAW), mode='r', encoding='utf-8') as reader:
            with codecs.open(vars.SNLI_TEST_MOD, 'w', encoding='utf-8') as writer:
                test_datalist = []

                for line in reader:
                    dic_line = json.loads(line)
                    label = dic_line['gold_label']
                    sentence1 = dic_line['sentence1']
                    sentence2 = dic_line['sentence2']

                    if label == 'neutral' or label == 'entailment':
                        writer.write(sentence1 + '\t' + sentence2 + '\n')

                    test_datalist.append([sentence1, sentence2, label])

        from collections import Counter

        with codecs.open(vars.SNLI_META, 'w', encoding='utf-8') as writer:
            writer.write('Train File Info:\n')
            writer.write('Total Samples: ' + str(len(train_datalist)) + '\n')
            writer.write('Total Different Sentences: ' + str(len(train_datalist) * 2) + '\n')
            writer.write('Total Neutral Sentences: ' + str(Counter([item[2] for item in train_datalist])['neutral']) + '\n')
            writer.write('Total Entailment Sentences: ' + str(Counter([item[2] for item in train_datalist])['entailment']) + '\n')
            writer.write('Total Contradiction Sentences: ' + str(Counter([item[2] for item in train_datalist])['contradiction']) + '\n')
            writer.write('Total Vocab Size: ' + str(
                len(set([word.replace("\"", "") for item in train_datalist for word in item[0].split()]
                        + [word.replace("\"", "") for item in train_datalist for word in item[1].split()]))))

            writer.write('\n\n')

            writer.write('DEV File Info:\n')
            writer.write('Total Samples: ' + str(len(dev_datalist)) + '\n')
            writer.write('Total Different Sentences: ' + str(len(dev_datalist) * 2) + '\n')
            writer.write(
                'Total Neutral Sentences: ' + str(Counter([item[2] for item in dev_datalist])['neutral']) + '\n')
            writer.write('Total Entailment Sentences: ' + str(
                Counter([item[2] for item in dev_datalist])['entailment']) + '\n')
            writer.write('Total Contradiction Sentences: ' + str(
                Counter([item[2] for item in dev_datalist])['contradiction']) + '\n')
            writer.write('Total Vocab Size: ' + str(
                len(set([word.replace("\"", "") for item in dev_datalist for word in item[0].split()]
                        + [word.replace("\"", "") for item in dev_datalist for word in item[1].split()]))))

            writer.write('\n\n')

            writer.write('Test File Info:\n')
            writer.write('Total Samples: ' + str(len(test_datalist)) + '\n')
            writer.write('Total Different Sentences: ' + str(len(test_datalist) * 2) + '\n')
            writer.write(
                'Total Neutral Sentences: ' + str(Counter([item[2] for item in test_datalist])['neutral']) + '\n')
            writer.write('Total Entailment Sentences: ' + str(
                Counter([item[2] for item in test_datalist])['entailment']) + '\n')
            writer.write('Total Contradiction Sentences: ' + str(
                Counter([item[2] for item in test_datalist])['contradiction']) + '\n')
            writer.write('Total Vocab Size: ' + str(
                len(set([word.replace("\"", "") for item in test_datalist for word in item[0].split()]
                        + [word.replace("\"", "") for item in test_datalist for word in item[1].split()]))))

            writer.write('\n')