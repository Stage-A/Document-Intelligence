import camelot
import pdfplumber
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        titles = self.get_table_titles(pdf_file)
        table_text = []
        for table, title in zip(tables, titles):
            # table_text.append((str(title) + "\n" + table.df.to_string()).replace("\\n", ""))
            table_text.append((table.df.to_string()).replace("\\n", ""))
        return table_text, titles
    
    def get_table_titles(self, pdf_file):
        with pdfplumber.open(pdf_file) as pdf:
            all_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
            pattern = r'ai_tables_#.*'
            matches = re.findall(pattern, all_text)
        return matches

class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)

class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        cosine = np.dot(vector_from_table, vector_from_keyword) / (norm(vector_from_table)*norm(vector_from_keyword))
        return cosine

def main(keyword, pdf_file):
    print("pdf content extracting....")
    pdf_parser = pdf2text()
    table_texts, titles = pdf_parser(pdf_file)

    print("text2vec converting....")
    text2vec_model = text2vector()
    table_vectors = []
    for table_text in table_texts:
        table_vectors.append(text2vec_model(table_text))
    
    print("cosine sim caculating....")
    cosine_sim_caculate = cosine_sim()
    key_vec = text2vec_model(keyword)
    table_sim = []
    for table_vector in table_vectors:
        table_sim.append(cosine_sim_caculate(table_vector, key_vec))
    
    print("\n" + "result table:")
    max_index = table_sim.index(max(table_sim))
    print(titles[max_index])
    print(table_texts[max_index])

if __name__ == "__main__":
    keyword = input("please input the keyword:")
    pdf_file = input("please input the pdf_file:")
    main(keyword, pdf_file)
    # main("keyword", "docs/1.pdf")
    # main("keyword", "docs/2.pdf")
