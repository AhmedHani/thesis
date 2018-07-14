import vars
import os
import shutil
import codecs


class PPDBDataset(object):

    def __init__(self, download_link=vars.PPDB_DOWNLOAD_LINK):
        if not self.__gzip_existed():
            self.__create_dir()
            self.__download_and_extract(download_link)

        if not self.__moddata_exist():
            self.__toformat()

    @staticmethod
    def __download_and_extract(download_link):
        import gzip, urllib.request, io

        url = download_link
        file_name = vars.PPDB_GZIP

        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

            compressed_file = io.BytesIO(response.read())
            decompressed_file = gzip.GzipFile(fileobj=compressed_file)

            with open(vars.PPDB_RAW, 'wb') as outfile:
                outfile.write(decompressed_file.read())

    @staticmethod
    def __gzip_existed():
        if os.path.exists(vars.PPDB_GZIP):
            return True

        return False

    @staticmethod
    def __create_dir():
        if os.path.exists(vars.PPDB_DIR):
            shutil.rmtree(vars.PPDB_DIR)

        os.mkdir(vars.PPDB_DIR)

    @staticmethod
    def __moddata_exist():
        if os.path.exists(vars.PPDB_MOD):
            return True

        return False

    @staticmethod
    def __toformat():
        datalist = []

        with codecs.open(vars.PPDB_RAW, 'r', encoding='utf-8') as reader:
            with codecs.open(vars.PPDB_MOD, 'w', encoding='utf-8') as writer:
                for line in reader:
                    line_tokens = line.strip().rstrip().split('|||')
                    sentence1 = line_tokens[1].rstrip().strip()
                    sentence2 = line_tokens[2].rstrip().strip()

                    writer.write(sentence1 + '\t' + sentence2 + '\n')
                    datalist.append([sentence1, sentence2])

        with codecs.open(vars.PPDB_META, 'w', encoding='utf-8') as writer:
            writer.write('Total Samples: ' + str(len(datalist)) + '\n')
            writer.write('Total Different Sentences: ' + str(len(datalist) * 2) + '\n')
            writer.write('Total Vocab Size: ' + str(
                len(set([word.replace("\"", "") for item in datalist for word in item[0].split()]
                        + [word.replace("\"", "") for item in datalist for word in item[1].split()]))))


