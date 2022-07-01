# Import dependencies
from collections import defaultdict
from heapq import nlargest
# from io import StringIO
from string import punctuation

import streamlit as st
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize

# If you have problems in install nltk, try the options:
#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

# import nltk
# import ssl
# try:
#      _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download()


# Functions for resume
stopwords_ptbr = set(stopwords.words('portuguese') + list(punctuation))


def remove_stopwords_and_punct_in_portuguese(text):
    words = word_tokenize(text.lower())
    return [word for word in words if word not in stopwords_ptbr]

def calculate_grade(sentences, important_words, distance):
  grades = []
  indice_text = 0

  for sentence in [nltk.word_tokenize(sentence.lower()) for sentence in sentences]:
    indice_word = []
    for word in important_words:
      try:
        indice_word.append(sentence.index(word))
      except ValueError:
        pass
    
    indice_word.sort()

    if len(indice_word) == 0:
      continue

    # [0, 1, 3, 5]
    list_groups = []
    group = [indice_word[0]]
    i = 1
    while i < len(indice_word):
      if indice_word[i] - indice_word[i - 1] < distance:
        group.append(indice_word[i])
      else:
        list_groups.append(group[:])
        group = [indice_word[i]]
      i += 1
    list_groups.append(group)

    grade_max_group = 0
    for g in list_groups:
      important_words_in_group = len(g)
      total_words_in_group = g[-1] - g[0] + 1
      grade = 1.0 *  important_words_in_group**2 / total_words_in_group

      if grade > grade_max_group:
        grade_max_group = grade

    grades.append((grade_max_group, indice_sentence))
    indice_sentence += 1

  return grades


def sumarize_text_freq(text, n_sent=10):
    words_not_stopwords = remove_stopwords_and_punct_in_portuguese(text)
    sentences = sent_tokenize(text)
    frequency = FreqDist(words_not_stopwords)
    important_sentences = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                important_sentences[i] += frequency[word]

    numb_sent = n_sent
    idx_important_sentences = nlargest(numb_sent,
                                       important_sentences,
                                       important_sentences.get)

    for i in sorted(idx_important_sentences):
        st.write(sentences[i])

def sumarize_text_luhn(text, n_sent=10):
    words_not_stopwords = remove_stopwords_and_punct_in_portuguese(text)
    sentences = sent_tokenize(text)
    frequency = FreqDist(words_not_stopwords)
    important_sentences = defaultdict(int)

    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                important_sentences[i] += frequency[word]

    numb_sent = n_sent
    idx_important_sentences = nlargest(numb_sent,
                                       important_sentences,
                                       important_sentences.get)

    for i in sorted(idx_important_sentences):
        st.write(sentences[i])



def main():
    st.title('Este é um experimento de Processamento de Linguagem Natural do TJSP')
    st.subheader('Resumo dos texto :')
    st.markdown("<h4 '>Entre com o texto original.</h4>", unsafe_allow_html=True)
    st.write('')

    uploaded_file = st.text_area('Cole aqui:',
                                 'O Tribunal de Justiça de São Paulo é considerado o maior tribunal do mundo em volume de\
                                  processos. O número de ações demandadas no Judiciário estadual paulista corresponde a 25%\
                                  do total de processos em andamento em toda a Justiça brasileira, incluindo cortes federais\
                                  e tribunais superiores (dados do relatório Justiça em Números 2020, produzido pelo Conselho\
                                  Nacional de Justiça). Consequentemente, é o tribunal com a maior força de trabalho: 2,5 mil\
                                  magistrados e aproximadamente 40 mil servidores, em 320 comarcas do Estado. Por ser um\
                                  Tribunal Estadual, tem como função julgar todas as causas que não se enquadram na competência\
                                  da Justiça especializada (Federal, do Trabalho, Eleitoral e Militar). Entre os tipos de demandas\
                                  recebidas na Justiça paulista estão a maioria das ações cíveis (indenizações, cobranças, Direito\
                                  do Consumidor, etc.); dos crimes comuns; processos das áreas de Família, Infância e Juventude,\
                                  Falências e Recuperações Judiciais e Registros Públicos; execuções fiscais dos Estados e\
                                  municípios etc. Por essa razão, a Justiça dos Estados é considerada a mais próxima do dia a dia\
                                  dos cidadãos.', height=300)

    st.subheader("Este é o resumo de seu texto :unlock:")

    if uploaded_file is not None:
        # To convert to a string based IO:
        # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # To read file as string:
        # uploaded_file = stringio.read()

        # Sidebar Menu
        options = ["Frequência de palavras", "Luhn"]
        menu = st.sidebar.selectbox("Escolha um algoritmo:", options)

        n_sent = st.sidebar.slider('Escolha o número de frases que deseja retornar:', value=1)


        # Choices
        if menu == "Frequência de palavras":
            sumarize_text_freq(uploaded_file, n_sent)

        if menu == "Luhn":
            sumarize_text_freq(uploaded_file, n_sent)

        st.sidebar.info('Check out the project on [Github](https://github.com/DanielaLFreire/NLP-Text-Summary-App)')


if __name__ == '__main__':
    main()
