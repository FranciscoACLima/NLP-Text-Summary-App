# Import dependencies
from collections import defaultdict
from heapq import nlargest
from string import punctuation

import os
import numpy as np
import networkx as nx

import streamlit as st

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance

import re
import spacy

from fpdf import FPDF
import base64

# sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.reduction import ReductionSummarizer

## Alternative Method using stopwords
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# If you have problems in install nltk, try the options:
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# !python -m spacy download pt_core_news_sm

# Parameters

stopwords_ptbr = set(stopwords.words('portuguese') + list(punctuation))
nlp = spacy.load("pt_core_news_sm")

path_theme = "./temas/texto_tema_1033_formatado.txt"
path_texts = "./acordaos"
path_summarize = "./resumos"


# Functions

def create_txt(text, name_txt):
    text2 = text.encode('latin-1', 'replace').decode('latin-1')
    with open(path_summarize + '/' + name_txt, 'w', encoding='utf-8') as f:
        for i in range(40):
            f.write(text2)
    return


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)
    return


def create_pdf(sentences, sentences_output, name):
    # pdf name
    name_pdf = name.replace('.txt', '.pdf')

    # Create instance of FPDF class
    #  'P'ortrait', A4 size paper, use mm as unit of measure
    pdf = FPDF('P', 'mm', 'A4')

    pdf.add_page()
    pdf.set_font("Arial", size=12)

    effective_page_width = pdf.w - 2 * pdf.l_margin

    # Set background color yellow, text 'J'ustified and allow filling the cell (fill=1)
    pdf.set_fill_color(251, 247, 25)

    for i in sentences:
        j = str(i).encode('latin-1', 'replace').decode('latin-1')

        if i in sentences_output:
            pdf.multi_cell(effective_page_width, 10, txt=j, fill=1, align='J')
            pdf.ln()
        else:
            pdf.multi_cell(effective_page_width, 10, txt=j, fill=0, align='J')
            pdf.ln()

    # save the pdf with name_pdf.pdf
    save_image_path = path_summarize + '/' + name_pdf
    pdf.output(save_image_path)

    return save_image_path


def read_file(path_name):
    path_name2 = path_name.encode('latin-1', 'replace').decode('latin-1')
    with open(path_name2, "r", encoding='utf-8') as f:
        ready_file = f.read()
    return ready_file


def file_selector(folder_path=path_texts):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Escolha o arquivo: ', filenames)
    return os.path.join(folder_path, selected_filename), selected_filename


# Preprocessing

def preprocessing(text):
    text = text.lower()
    text = re.sub(r" +", ' ', text)

    # more clean
    text = text.replace('.', '')
    text = text.replace('/', ' ')
    text = text.replace('-', ' ')
    text = text.replace(',', '')

    document = nlp(text)
    tokens = []
    for token in document:
        tokens.append(token.lemma_)

    tokens = [word for word in tokens if word not in stopwords_ptbr]  # and word.isalpha()]
    formatted_text = ' '.join([str(element) for element in tokens])  # if not element.isdigit()])

    return formatted_text


# option_1: summarize by keywords

def theme_search(text, path):
    # import theme text
    text_theme = read_file(path)

    # test if the word is in theme text
    sentences = sent_tokenize(text)
    sentences_theme = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for word in word_tokenize(preprocessing(sentence)):
            if word in word_tokenize(text_theme):
                if i not in sentences_theme.keys():
                    sentences_theme[i] = sentence

    return sentences_theme


# option_2: algorithm - menu = "Frequência de palavras"

def sumarize_text_freq(text, n_sent=10):
    # call preprocessing
    formatted_text = preprocessing(text)
    sentences = sent_tokenize(text)
    frequency = FreqDist(word_tokenize(formatted_text))
    max_frequency = max(frequency.values())
    important_sentences = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                important_sentences[i] += (frequency[word] / max_frequency)

    numb_sent = n_sent
    idx_important_sentences = nlargest(numb_sent,
                                       important_sentences,
                                       important_sentences.get)

    sentences_alg = defaultdict(int)

    for i in sorted(idx_important_sentences):
        sentences_alg[i] = sentences[i]

    return sentences_alg


# option_2: algorithm - menu = "Luhn"

def grade_calculate(sentences, important_words, distance):
    grades = []
    index_text = 0

    for sentence in [word_tokenize(sentence.lower()) for sentence in sentences]:
        index_word = []
        for word in important_words:
            try:
                index_word.append(sentence.index(word))
            except ValueError:
                pass

        index_word.sort()

        if len(index_word) == 0:
            continue

        list_groups = []
        group = [index_word[0]]
        i = 1
        while i < len(index_word):
            if index_word[i] - index_word[i - 1] < distance:
                group.append(index_word[i])
            else:
                list_groups.append(group[:])
                group = [index_word[i]]
            i += 1
        list_groups.append(group)

        grade_max_group = 0
        for g in list_groups:
            important_words_in_group = len(g)
            total_words_in_group = g[-1] - g[0] + 1
            grade = 1.0 * important_words_in_group ** 2 / total_words_in_group

            if grade > grade_max_group:
                grade_max_group = grade

        grades.append((grade_max_group, index_text))
        index_text += 1

    return grades


def sumarize_text_luhn(sentences, formatted_text, n_sent, top_n_words=100, distance=5):
    words = [word for sentence in formatted_text for word in word_tokenize(sentence)]
    frequency = FreqDist(words)

    top_n_words = [word[0] for word in frequency.most_common(top_n_words)]
    sentences_grade = grade_calculate(formatted_text, top_n_words, distance)

    idx_important_sentences = nlargest(n_sent, sentences_grade)
    sentences_alg = defaultdict(int)

    for (grade, i) in sorted(idx_important_sentences):
        sentences_alg[i] = sentences[i]

    return sentences_alg


# option_2: algorithm - menu = "PageRank"

def similar_calculate(sent1, sent2):
    words1 = [word for word in word_tokenize(sent1)]
    words2 = [word for word in word_tokenize(sent2)]

    all_words = list(set(words1 + words2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in words1:
        vector1[all_words.index(word)] += 1
    for word in words2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)


def matrix_calculate(sentences):
    matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            matrix[i][j] = similar_calculate(sentences[i], sentences[j])

    return matrix


def sumarize_text_pagerank(sentences, formatted_text, n_sent):
    matrix = matrix_calculate(formatted_text)

    graph = nx.from_numpy_array(matrix)

    grades = nx.pagerank(graph)

    order_grades = sorted(((grades[i], grade, i) for i, grade in enumerate(sentences)), reverse=True)

    idx_important_sentences = nlargest(n_sent, order_grades)
    sentences_alg = defaultdict(int)

    for (grade, sent, i) in idx_important_sentences:
        sentences_alg[i] = sent

    return sentences_alg


# option_2: algorithm - menu = "Edmundson"

def sumarize_text_Edm(path_theme, sentences, text, n_sent):
    # import theme text
    text_theme = read_file(path_theme)

    parser = PlaintextParser.from_string(text, Tokenizer('portuguese'))
    sumarizer_Edm = EdmundsonSummarizer()
    sumarizer_Edm = LsaSummarizer(Stemmer("portuguese"))
    sumarizer_Edm.stop_words = get_stop_words("portuguese")
    sumarizer_Edm.bonus_words = ('tribunal', 'STJ')
    # [word for word in word_tokenize(text_theme)]
    sumarizer_Edm.stigma_words = 'de'
    sumarizer_Edm.null_words = 'de'

    idx_important_sentences = sumarizer_Edm(parser.document, n_sent)

    sentences_alg = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for idx_sent in idx_important_sentences:
            if str(idx_sent) == sentence:
                sentences_alg[i] = sentence

    return sentences_alg


# option_2: algorithm - menu = "LSA (Latent Semantic Analysis)

def sumarize_text_LSA(sentences, text, n_sent):
    parser = PlaintextParser.from_string(text, Tokenizer('portuguese'))
    sumarizer_Lsa = LsaSummarizer()
    sumarizer_Lsa = LsaSummarizer(Stemmer("portuguese"))
    sumarizer_Lsa.stop_words = get_stop_words("portuguese")

    idx_important_sentences = sumarizer_Lsa(parser.document, n_sent)

    sentences_alg = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for idx_sent in idx_important_sentences:
            if str(idx_sent) == sentence:
                sentences_alg[i] = sentence

    return sentences_alg


# option_2: algorithm - menu = "LexRank"

def sumarize_text_LR(sentences, text, n_sent):
    parser = PlaintextParser.from_string(text, Tokenizer('portuguese'))
    sumarizer_LR = LexRankSummarizer()
    sumarizer_LR = LexRankSummarizer(Stemmer("portuguese"))
    sumarizer_LR.stop_words = get_stop_words("portuguese")

    idx_important_sentences = sumarizer_LR(parser.document, n_sent)

    sentences_alg = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for idx_sent in idx_important_sentences:
            if str(idx_sent) == sentence:
                sentences_alg[i] = sentence

    return sentences_alg


# option_2: algorithm - menu = "TextRank"

def sumarize_text_TR(sentences, text, n_sent):
    parser = PlaintextParser.from_string(text, Tokenizer('portuguese'))
    sumarizer_TR = TextRankSummarizer()
    sumarizer_TR = TextRankSummarizer(Stemmer("portuguese"))
    sumarizer_TR.stop_words = get_stop_words("portuguese")

    idx_important_sentences = sumarizer_TR(parser.document, n_sent)

    sentences_alg = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for idx_sent in idx_important_sentences:
            if str(idx_sent) == sentence:
                sentences_alg[i] = sentence

    return sentences_alg


# option_2: algorithm - menu = "KL-Soma"

def sumarize_text_KL(sentences, text, n_sent):
    parser = PlaintextParser.from_string(text, Tokenizer('portuguese'))
    sumarizer_KL = KLSummarizer()
    sumarizer_KL = KLSummarizer(Stemmer("portuguese"))
    sumarizer_KL.stop_words = get_stop_words("portuguese")

    idx_important_sentences = sumarizer_KL(parser.document, n_sent)

    sentences_alg = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for idx_sent in idx_important_sentences:
            if str(idx_sent) == sentence:
                sentences_alg[i] = sentence

    return sentences_alg


# option_2: algorithm - menu = "Redução"

def sumarize_text_Red(sentences, text, n_sent):
    parser = PlaintextParser.from_string(text, Tokenizer('portuguese'))
    sumarizer_Red = ReductionSummarizer()
    sumarizer_Red = ReductionSummarizer(Stemmer("portuguese"))
    sumarizer_Red.stop_words = get_stop_words("portuguese")

    idx_important_sentences = sumarizer_Red(parser.document, n_sent)

    sentences_alg = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for idx_sent in idx_important_sentences:
            if str(idx_sent) == sentence:
                sentences_alg[i] = sentence

    return sentences_alg


# main

def main():
    # st.image("logo.png", width=150)
    st.title('Experimento Inicial do Projeto TJSP')
    st.markdown("<h4 '>Sumarização de Texto com Processamento de Linguagem Natural</h4>", unsafe_allow_html=True)

    file_path, file_name = file_selector()

    try:
        with open(file_path, 'r') as input:
            myfile = input.read()
    except FileNotFoundError:
        st.error('Arquivo não pode ser aberto.')

    st.subheader('Texto original')

    uploaded_file = st.text_area('Conteúdo do arquivo: ', value=myfile, height=100)

    st.subheader("Resumo")

    if uploaded_file is not None:

        # Sidebar Menu
        st.sidebar.image("./images/logo.png", width=150)

        # How will the search be
        st.sidebar.write('Escolha como deseja que seu texto seja resumido:')
        option_1 = st.sidebar.checkbox("por palavras-chaves")
        # if option_1:
        # st.sidebar.write("[Ver](https://github.com/DanielaLFreire/NLP-Text-Summary-App/blob/master/temas"
        #                 "/texto_tema_1033_formatado.txt)")
        option_2 = st.sidebar.checkbox("por algoritmo")

        sentences = [sentence for sentence in sent_tokenize(uploaded_file)]
        formatted_text = [preprocessing(sentence) for sentence in sentences]

        # compression rate is the amount that can be extracted from the text for the abstract, according to
        # Pardo[1]\ it must be between 20 to 50% of the text to generate an efficient summary.
        len_file = len(sentences)
        n_sent_default = len_file * 0.35
        n_sent_default = int(n_sent_default)
        n_sent = st.sidebar.slider(label='Escolha o número de frases que deseja retornar:',
                                   min_value=1, max_value=len_file, value=n_sent_default)

        sentences_output = {}
        sentences_output2 = {}
        sentences_committee = {}

        if option_1:
            sentences_theme = theme_search(uploaded_file, path_theme)

            for i in sorted(sentences_theme):
                sentences_output[i] = sentences_theme[i]

        if option_2:

            # Which will the algorithm be
            algorithms = ["Comitê(todos)", "Frequência de palavras", "Luhn", "PageRank", "Edmundson",
                          "LSA (Latent Semantic Analysis)",
                          "LexRank", "TextRank", "KL-Soma", "Redução"]
            menu = st.sidebar.selectbox("Escolha um algoritmo:", algorithms)

            # Choices
            if menu == "Frequência de palavras" or menu == "Comitê(todos)":
                sentences_text = sumarize_text_freq(uploaded_file, n_sent)
                if "Comitê(todos)":
                    for i in sorted(sentences_text):
                        if i not in sentences_committee.keys():
                            sentences_committee[i] = 1, sentences_text[i]
                        else:
                            sentences_committee[i] = sentences_committee[i][0] + 1, sentences_committee[i][1]

            if menu == "Luhn" or menu == "Comitê(todos)":
                sentences_text = sumarize_text_luhn(sentences, formatted_text, n_sent, 100, 5)
                if "Comitê(todos)":
                    for i in sorted(sentences_text):
                        if i not in sentences_committee.keys():
                            sentences_committee[i] = 1, sentences_text[i]
                        else:
                            sentences_committee[i] = sentences_committee[i][0] + 1, sentences_committee[i][1]

            if menu == "PageRank" or menu == "Comitê(todos)":
                sentences_text = sumarize_text_pagerank(sentences, formatted_text, n_sent)
                if "Comitê(todos)":
                    for i in sorted(sentences_text):
                        if i not in sentences_committee.keys():
                            sentences_committee[i] = 1, sentences_text[i]
                        else:
                            sentences_committee[i] = sentences_committee[i][0] + 1, sentences_committee[i][1]

            if menu == "Edmundson" or menu == "Comitê(todos)":
                sentences_text = sumarize_text_Edm(path_theme, sentences, uploaded_file, n_sent)
                if "Comitê(todos)":
                    for i in sorted(sentences_text):
                        if i not in sentences_committee.keys():
                            sentences_committee[i] = 1, sentences_text[i]
                        else:
                            sentences_committee[i] = sentences_committee[i][0] + 1, sentences_committee[i][1]

            if menu == "LSA (Latent Semantic Analysis)" or menu == "Comitê(todos)":
                sentences_text = sumarize_text_LSA(sentences, uploaded_file, n_sent)
                if "Comitê(todos)":
                    for i in sorted(sentences_text):
                        if i not in sentences_committee.keys():
                            sentences_committee[i] = 1, sentences_text[i]
                        else:
                            sentences_committee[i] = sentences_committee[i][0] + 1, sentences_committee[i][1]

            if menu == "LexRank" or menu == "Comitê(todos)":
                sentences_text = sumarize_text_LR(sentences, uploaded_file, n_sent)
                if "Comitê(todos)":
                    for i in sorted(sentences_text):
                        if i not in sentences_committee.keys():
                            sentences_committee[i] = 1, sentences_text[i]
                        else:
                            sentences_committee[i] = sentences_committee[i][0] + 1, sentences_committee[i][1]

            if menu == "TextRank" or menu == "Comitê(todos)":
                sentences_text = sumarize_text_TR(sentences, uploaded_file, n_sent)
                if "Comitê(todos)":
                    for i in sorted(sentences_text):
                        if i not in sentences_committee.keys():
                            sentences_committee[i] = 1, sentences_text[i]
                        else:
                            sentences_committee[i] = sentences_committee[i][0] + 1, sentences_committee[i][1]

            if menu == "KL-Soma" or menu == "Comitê(todos)":
                sentences_text = sumarize_text_KL(sentences, uploaded_file, n_sent)
                if "Comitê(todos)":
                    for i in sorted(sentences_text):
                        if i not in sentences_committee.keys():
                            sentences_committee[i] = 1, sentences_text[i]
                        else:
                            sentences_committee[i] = sentences_committee[i][0] + 1, sentences_committee[i][1]

            if menu == "Redução" or menu == "Comitê(todos)":
                sentences_text = sumarize_text_Red(sentences, uploaded_file, n_sent)
                if "Comitê(todos)":
                    for i in sorted(sentences_text):
                        if i not in sentences_committee.keys():
                            sentences_committee[i] = 1, sentences_text[i]
                        else:
                            sentences_committee[i] = sentences_committee[i][0] + 1, sentences_committee[i][1]

            if menu == "Comitê(todos)":
                for i in sorted(sentences_committee, key=sentences_committee.get, reverse=True):
                    sentences_output[i] = sentences_committee[i][1]
            else:
                for i in sorted(sentences_text):
                    sentences_output[i] = sentences_text[i]

        for i in sorted(sentences_output)[:n_sent]:
            sentences_output2[i] = sentences_output[i]
        summary = ' '.join(sentences_output[i] for i in sorted(sentences_output2)[:n_sent])

        st.text_area('Conteúdo resumido: ', value=summary, height=100)

        if not st.button(label="📥 Salve seu resumo!"):
            pass
        else:
            save_image_path = create_pdf(sentences, sentences_output2.values(), 'Resumo_' + file_name)
            if save_image_path is not None:
                st.success("Resumo salvo com sucesso.")
                show_pdf(save_image_path)
        if option_2:
            expander = st.expander("Deixe sua avaliação.")
            expander.radio('O que você achou do algoritmo "' + menu + '":', ('Ruim', 'Regular', 'Bom'), horizontal=True)

    st.sidebar.info('Confira o projeto no [Github](https://github.com/DanielaLFreire/NLP-Text-Summary-App)')


if __name__ == '__main__':
    main()
