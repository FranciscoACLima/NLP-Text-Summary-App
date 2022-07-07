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

# If you have problems in install nltk, try the options:
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# !python -m spacy download pt_core_news_sm


# Functions for resume
stopwords_ptbr = set(stopwords.words('portuguese') + list(punctuation))
nlp = spacy.load("pt_core_news_sm")

path_theme = "./temas/texto_tema_1033_formatado.txt"
path_texts = "./acordaos"


def theme_search(text, path):
    # import theme text
    with open(path, 'r') as f:
        text_theme = f.read()

    # test if the word is in theme text
    sentences = sent_tokenize(text)
    sentences_theme = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for word in word_tokenize(preprocessing(sentence)):
            if word in word_tokenize(text_theme):
                if i not in sentences_theme.keys():
                    sentences_theme[i] = sentence

    return sentences_theme


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


###FREQUENCIA_PALAVRAS###
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


###LUHN###
def sumarize_text_luhn(text, n_sent=10, top_n_words=100, distance=5):
    sentences = [sentence for sentence in sent_tokenize(text)]
    formatted_text = [preprocessing(sentence) for sentence in sentences]
    words = [word for sentence in formatted_text for word in word_tokenize(sentence)]
    frequency = FreqDist(words)

    top_n_words = [word[0] for word in frequency.most_common(top_n_words)]
    sentences_grade = grade_calculate(formatted_text, top_n_words, distance)

    idx_important_sentences = nlargest(n_sent, sentences_grade)
    sentences_alg = defaultdict(int)

    for (grade, i) in sorted(idx_important_sentences):
        sentences_alg[i] = sentences[i]

    return sentences_alg


###PAGERANK###
def sumarize_text_pagerank(text, n_sent=10):
    sentences = [sentence for sentence in sent_tokenize(text)]
    formatted_text = [preprocessing(sentence) for sentence in sentences]

    matrix = matrix_calculate(formatted_text)

    graph = nx.from_numpy_array(matrix)

    grades = nx.pagerank_numpy(graph)

    order_grades = sorted(((grades[i], grade, i) for i, grade in enumerate(sentences)), reverse=True)

    idx_important_sentences = nlargest(n_sent, order_grades)
    sentences_alg = defaultdict(int)

    for (grade, sent, i) in idx_important_sentences:
        sentences_alg[i] = sent

    return sentences_alg


def file_selector(folder_path=path_texts):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Escolha o arquivo: ', filenames)
    return os.path.join(folder_path, selected_filename), selected_filename


def main():
    # st.image("logo.png", width=150)
    st.title('Experimento de Processamento de Linguagem Natural do TJSP')

    f_path, f = file_selector()

    try:
        with open(f_path, 'r') as input:
            myfile = input.read()
    except FileNotFoundError:
        st.error('Arquivo não pode ser aberto.')

    st.subheader('Texto original')
    # st.markdown("<h4 '>Entre com o texto original.</h4>", unsafe_allow_html=True)

    uploaded_file = st.text_area('Conteúdo do arquivo: ', value=myfile, height=100)

    st.subheader("Resumo :")

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

        sentences_output = {}

        if option_1:
            sentences_theme = theme_search(uploaded_file, path_theme)

            for i in sorted(sentences_theme):
                sentences_output[i] = sentences_theme[i]

        if option_2:
            n_sent = st.sidebar.slider('Escolha o número de frases que deseja retornar:', value=10)
            # Which will the algorithm be
            algorithms = ["Frequência de palavras", "Luhn", "PageRank"]
            menu = st.sidebar.selectbox("Escolha um algoritmo:", algorithms)


            # Choices
            if menu == "Frequência de palavras":
                sentences_text = sumarize_text_freq(uploaded_file, n_sent)

            if menu == "Luhn":
                sentences_text = sumarize_text_luhn(uploaded_file, n_sent, 100, 5)

            if menu == "PageRank":
                sentences_text = sumarize_text_pagerank(uploaded_file, n_sent)

            for i in sorted(sentences_text):
                sentences_output[i] = sentences_text[i]


        mysummary = ' '.join(sentences_output[i] for i in sorted(sentences_output))

        st.text_area('Conteúdo resumido: ', value=mysummary, height=100)

        st.download_button(label="📥 Salve seu resumo!", data=mysummary, file_name="Resumo.txt" )

        if option_2:
            def radio_changed():
                expander.write('**Agradecemos por sua opinião!**')

            expander = st.expander("Deixe sua avaliação.")
            expander.radio('O que você achou do algoritmo "' + menu + '":', ('Ruim', 'Regular', 'Bom'), on_change=radio_changed, horizontal=True)


    st.sidebar.info('Confira o projeto no [Github](https://github.com/DanielaLFreire/NLP-Text-Summary-App)')


if __name__ == '__main__':
    main()
