import vars
import os
import shutil
import codecs


class QuoraQuestionsPairDataset(object):

    def __init__(self, download_link=vars.QUORA_QUESTIONS_PAIRS_DOWNLOAD_LINK):
        if not self.__rawdata_existed():
            self.__create_dir()
            self.__download(download_link)

        if not self.__moddata_existed():
            self.__toformat()

    @staticmethod
    def __download(download_link):
        from urllib.request import urlopen

        file_name = download_link.split('/')[-1]
        u = urlopen(download_link)
        f = open(vars.QUORA_QUESTIONS_PAIRS_RAW, 'wb')
        meta = u.info()
        file_size = int(meta._headers[-2][1])

        print("Downloading: %s Bytes: %s" % (file_name, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8) * (len(status) + 1)
            print(status.replace('\b', ''))

        f.close()

    @staticmethod
    def __rawdata_existed():
        if os.path.exists(vars.QUORA_QUESTIONS_PAIRS_RAW):
            return True

        return False

    @staticmethod
    def __moddata_existed():
        if os.path.exists(vars.QUORA_QUESTIONS_PAIRS_MOD):
            return True

        return False

    @staticmethod
    def __create_dir():
        if os.path.exists(vars.QUORA_QUESTIONS_PAIRS_DIR):
            shutil.rmtree(vars.QUORA_QUESTIONS_PAIRS_DIR)

        os.mkdir(vars.QUORA_QUESTIONS_PAIRS_DIR)

    @staticmethod
    def __toformat():
        datalist = []

        with codecs.open(filename=vars.QUORA_QUESTIONS_PAIRS_RAW,
                         mode='r',
                         encoding='utf-8') as reader:
            with codecs.open(filename=vars.QUORA_QUESTIONS_PAIRS_MOD,
                             mode='w',
                             encoding='utf-8') as writer:
                for line in reader:
                    line_tokens = line.strip().rstrip().split('\t')
                    interest = line_tokens[3:]

                    try:
                        question1 = interest[0]
                        question2 = interest[1]
                        paraphrase = interest[2]

                        if int(paraphrase) == 1:
                            writer.write(question1 + '\t' + question2 + '\n')

                        datalist.append([question1, question2, int(paraphrase)])
                    except:
                        continue

        from collections import Counter

        with codecs.open(filename=vars.QUORA_QUESTIONS_PAIRS_META,
                         mode='w',
                         encoding='utf-8') as writer:
            writer.write('Total Samples: ' + str(len(datalist)) + '\n')
            writer.write('Total Different Sentences: ' + str(len(datalist) * 2) + '\n')
            writer.write('Total Non-Paraphrase: ' + str(Counter([item[2] for item in datalist])[0]) + '\n')
            writer.write('Total Paraphrase: ' + str(Counter([item[2] for item in datalist])[1]) + '\n')
            writer.write('Total Vocab Size: ' + str(len(set([word.replace("\"", "") for item in datalist for word in item[0].split()]
                                                            + [word.replace("\"", "") for item in datalist for word in item[1].split()]))))
            writer.write("\n")