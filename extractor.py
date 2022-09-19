""" Основной модуль выделения цитат """
import re

from typing import Union

import pymorphy2
import spacy
import torch

from navec import Navec
from slovnet.model.emb import NavecEmbedding
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    NamesExtractor,
    Doc,
)
from natasha.doc import DocSent


# Создание предиктора с кореференцией
predictor = spacy.load("ru_core_news_lg")
predictor.add_pipe("xx_coref")

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)

# Специльный символ, на который будет заменяться цитата. Таким образом,
# будет упрощена грамматика в предложении
SPEC = "¶"

# Путь к модели
PATH = "navec_news_v1_1B_250K_300d_100q.tar"

# Возможные варианты
SAID_SOMEONE = r'([“”"\'«][^\x22“”"\'«]+["\'»“”]\s{0,1},\s{0,1}[—-])'
SOMEONE_SAID = r'(:\s{0,1}[“”"\'«][^\x22“”"\'«]+?["\'»“”])'

# Объединяем все взможные варианты
ALL_QUOTES = f"{SAID_SOMEONE}|{SOMEONE_SAID}"

# Ищет слова с кавычками "интерфакс"
ONLY_QUOTATION_MARK = r'(([“”"\'«])([а-яА-яa-zA-Z0-9 - -\(/\)]*)(["\'»“”]))'


def get_similarity(word1: str, word2: str) -> float:
    """
    Получение сходства слов (cousine similarity)
    :return: Если вероятность больше заданной, вероятность. Иначе 0
    """
    probability = 0.02

    navec = Navec.load(PATH)  # ~1 sec, ~100MB RAM

    words = [word1]
    ids = [navec.vocab[_] for _ in words]
    emb = NavecEmbedding(navec)
    input_ = torch.tensor(ids)
    emb1 = emb(input_)

    words = [word2]
    ids = [navec.vocab[_] for _ in words]
    emb = NavecEmbedding(navec)
    input_ = torch.tensor(ids)
    emb2 = emb(input_)
    score = torch.cosine_similarity(emb1, emb2)[0]
    if score >= probability:
        return score
    return 0


def get_norm(word: str) -> str:
    """
    Получение нормальной формы слова
    :param word: Слово для приведения в нормальную форму
    :return:     Слово в нормальной форме
    """
    morph = pymorphy2.MorphAnalyzer(lang="ru")
    return morph.parse(word)[0].normal_form


def get_id(sentence: DocSent, person: str, id: str, root_id):
    """
    Рекурсия для получения слова, на которое ссылается глагол
    :param sentence: Предложение, где ищем
    :param person:   Имя персоны
    :param id:       Id слова
    :param root_id:  Id токена, на которое ссылается цитата
    :return:
    """
    if id == root_id:
        return person
    if id.rpartition("_")[2] == "0":
        # Если не нашли в предложении токен, на который ссылается
        return False
    # Присваиваем id число, на которое ссылается токен
    id = sentence[int(id.rpartition("_")[2]) - 1].head_id
    return get_id(sentence, person, id, root_id)


# Типы, которые могут присутствовать у слова
TYPES_ = ["parataxis", "root", "nmod"]
PERSONS_ = ["PER", "ORG"]


def get_real_person(num: int, fake_verb: str, doc: Doc) -> Union[str, None]:
    """

    :param doc:       Документ новости (объект natasha)
    :param num:       Номер предложения, где может быть персона
    :param fake_verb: Глагол, который ссылается на цитату
    :return:          Имя в случае удачи, иначе None
    """
    # TODO: Сделать, что бы возвращались имена через наташу
    head_id = None

    text = doc.sents[num]
    candidates = [x for x in doc.sents[num].spans if x.type in PERSONS_]
    if candidates:
        candidate = candidates[0].text
        head_id = candidates[0].tokens[0].head_id

    # Проверим, не сложное ли составное имя
    for x in text.tokens:
        if x.id == head_id:
            if x.rel in ["appos"]:
                head_id = x.head_id

    # Нашли, на что ссылается
    for x in text.tokens:
        if x.id == head_id:
            if x.rel in ["nsubj", "nsubj:pass"]:
                head_id = x.head_id

    second = None
    # Найдем слово, на которое ссылается
    for i in text.tokens:
        if i.head_id == head_id and i.rel == "obl":
            second = i.text
            second = get_similarity(get_norm(fake_verb), get_norm(second))
        if i.id == head_id and i.rel in TYPES_:
            score = get_similarity(get_norm(fake_verb), get_norm(i.text))
            if score > 0.12:
                if second:
                    if (second * 0.5 + score * 0.5) > 0.1:
                        return candidate
                else:
                    return candidate
            else:
                return None


def get_person(num: int, text: DocSent, doc: Doc) -> Union[str, None]:
    """
    Полуение имени говорившего цитату
    :param doc:     Документ новости (объект natasha)
    :param num:     Номер предложения, где встретилась цитата
    :param text:    Объект предложения Natasha
    :return:        Персона, говорившая цитату. В случае неудачи None
    """
    root_id = None
    # Найдем id слова, на которое ссылается цитата
    for i in text.tokens:
        if i.text == SPEC:
            root_id = i.head_id
            break

    # Если персона уже есть в предложении
    if text.spans:
        for i in text.spans:
            if i.type == "PER":
                # В случае, если одно имя
                if i.tokens[0].head_id == root_id:
                    return i.text
                # Иначе получим id, на которое ссылается
                return get_id(text.tokens, i.text, i.tokens[0].head_id, root_id)

    # TODO может есть просто обращение nmod

    if root_id:
        for i in text.tokens:
            if i.head_id == root_id and i.rel == "nsubj":
                return i.text
            if i.head_id == root_id and i.rel == "obl":
                # Если это сложное предложение, то найдем говорящего
                if num > 0:
                    return get_real_person(num - 1, i.text, doc)
                return get_real_person(num + 1, i.text, doc)
    return None


# Предобработка текста
def get_clear(text: str) -> [list, str]:
    """
    Предобработка текста. Поиск цитат
    :param text: Текст
    :return:
    """
    # Проход по всем кавычкам, удаление их из текста
    for i in re.findall(ONLY_QUOTATION_MARK, text):
        text = re.sub(i[0], i[0][1:-1], text)

    quotes = None
    if re.compile(ALL_QUOTES).search(text):
        # Учитывая, что у нас два паттерна, то вернется 1, если он есть
        quotes = [
            re.search(r'([«"](.+)["»])', [quote for quote in searched if quote][0])[0]
            for searched in re.compile(ALL_QUOTES).findall(text)
        ]
        text = re.sub(ALL_QUOTES, f"{SPEC}", text)
    return quotes, text


def get_doc(text: str) -> Doc:
    """
    Получение текста с разметкой через natasha
    :param text: Текст
    :return:
    """
    try:
        doc = predictor(text)
        text = doc._.resolved_text
    except ValueError:
        pass
    except IndexError:
        pass

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    doc.parse_syntax(syntax_parser)
    return doc


# Главный запуск
def get_quotes(text: str) -> Union[None, list]:
    """
    Главная функция получения цитат
    :param text: Текст с удаленным спец символом и
    :return:     Цитаты
    """
    quotes, text = get_clear(text)
    # Составляем список словарей, т.к. может быть более одной цитаты для персоны в тексте
    if quotes:
        doc = get_doc(text)
        result = []
        for _, sentence in enumerate(doc.sents):
            if SPEC in sentence.text:
                result.append({get_person(_, sentence, doc): quotes.pop(0)})
        return result
    return None
