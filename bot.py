import spacy
import wikipedia
import logging
import sys
import os
from telegram.ext import (
    Updater, CommandHandler, MessageHandler, Filters,
    ConversationHandler
)
from clarifai.client.model import Model

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Загрузка моделей NLP
nlp = spacy.load('en_core_web_md')  # Используем более полную модель из wiki_bot
nlp_sm = spacy.load('en_core_web_sm')  # Для pizza_bot

TOKEN = "YOUR_TELEGRAM_TOKEN_HERE"
CLARIFAI_API_KEY = "YOUR_CLARIFAI_KEY_HERE"

# ==================== ФУНКЦИИ ИЗ WIKI_BOT ====================

# Извлечение ключевой фразы из текста
def keyphrase(doc):
    # 1. Поиск предложного дополнения (после of, about..)
    for t in doc:
        if t.dep_ == 'pobj' and t.pos_ in ['NOUN', 'PROPN', 'VERB']:
            left_mods = []
            for child in t.lefts:
                if child.dep_ in ['amod', 'compound', 'nummod'] or child.pos_ in ['ADJ', 'NOUN', 'NUM']:
                    if child.pos_ != 'DET':
                        left_mods.append(child.text)
            
            phrase = (' '.join(left_mods) + ' ' + t.text).lstrip()

            preps_from_pobj = []
            for child in t.children:
                if child.dep_ == 'prep':
                    for prep_child in child.children:
                        if prep_child.dep_ == 'pobj' or (prep_child.pos_ in ['NOUN', 'PROPN'] and prep_child.head == child):
                            child_mods = []
                            for subchild in prep_child.children:
                                if subchild.dep_ in ['amod', 'compound'] and subchild.pos_ != 'DET':
                                    child_mods.append(subchild.text)
                            
                            if child_mods:
                                preps_from_pobj.append(child.text + ' ' + ' '.join(child_mods) + ' ' + prep_child.text)
                            else:
                                preps_from_pobj.append(child.text + ' ' + prep_child.text)
                            break
            
            prep_parent = t.head
            verb_parent = prep_parent.head
            
            preps_from_verb = []
            for child in verb_parent.children:
                if child.dep_ == 'prep':
                    if child == prep_parent:
                        continue
                    
                    for prep_child in child.children:
                        if prep_child.dep_ == 'pobj' or (prep_child.pos_ in ['NOUN', 'PROPN'] and prep_child.head == child):
                            child_mods = []
                            for subchild in prep_child.children:
                                if subchild.dep_ in ['amod', 'compound'] and subchild.pos_ != 'DET':
                                    child_mods.append(subchild.text)
                            
                            if child_mods:
                                preps_from_verb.append(child.text + ' ' + ' '.join(child_mods) + ' ' + prep_child.text)
                            else:
                                preps_from_verb.append(child.text + ' ' + prep_child.text)
                            break
            
            all_preps = preps_from_pobj + preps_from_verb
            if all_preps:
                phrase = phrase + ' ' + ' '.join(all_preps)
            
            return phrase
    
    # 2. Обработка придаточного дополнения
    for t in doc:
        if t.dep_ == 'ccomp' and t.pos_ in ['NOUN', 'VERB']:
            subject_parts = []
            
            for child in t.children:
                if child.dep_ in ['nsubj', 'compound', 'amod'] and child.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                    if child.dep_ == 'amod':
                        subject_parts.append(child.text)
                    elif child.dep_ == 'compound':
                        subject_parts.append(child.text)
                    elif child.dep_ == 'nsubj':
                        for left_child in child.lefts:
                            if left_child.dep_ in ['amod', 'compound'] and left_child.pos_ != 'DET':
                                subject_parts.append(left_child.text)
                        subject_parts.append(child.text)
            
            if subject_parts:
                unique_parts = []
                for part in subject_parts:
                    if part not in unique_parts:
                        unique_parts.append(part)
                
                subject = ' '.join(unique_parts)
                verb_lemma = t.lemma_
                phrase = subject + ' ' + verb_lemma
                
                for child in t.children:
                    if child.dep_ == 'dobj' and child.pos_ in ['NOUN', 'PROPN', 'VERB']:
                        phrase = phrase + ' ' + child.text
                        break
                
                return phrase

    # 3. Поиск подлежащего и глагола
    for t in reversed(doc):
        if (t.dep_ in ['nsubj', 'compound']) and t.pos_ in ['NOUN', 'PROPN']:
            noun_group_head = t
            while noun_group_head.head.pos_ in ['NOUN', 'PROPN'] and noun_group_head.head != noun_group_head:
                noun_group_head = noun_group_head.head
            
            verb = noun_group_head.head
            
            if verb.lemma_.lower() in ['tell', 'show', 'explain', 'ask']:
                for token in doc:
                    if token.pos_ == 'VERB' and token.lemma_.lower() not in ['tell', 'show', 'explain', 'ask']:
                        verb = token
                        break
            
            subject_parts = []
            
            def collect_noun_group(token, collected):
                for child in token.children:
                    if child.dep_ == 'amod' and child.pos_ == 'ADJ':
                        if child.text not in collected:
                            collected.insert(0, child.text)
                
                for child in token.children:
                    if child.dep_ == 'compound' and child.pos_ in ['NOUN', 'PROPN']:
                        if child.text not in collected:
                            collected.insert(0, child.text)
                
                if token.text not in collected:
                    collected.append(token.text)
                
                if token.head != token and token.head.pos_ in ['NOUN', 'PROPN']:
                    collect_noun_group(token.head, collected)
            
            collect_noun_group(t, subject_parts)
            subject = ' '.join(subject_parts)
            verb_lemma = verb.lemma_
            phrase = subject + ' ' + verb_lemma
            
            dobj_found = False
            for child in verb.children:
                if child.dep_ == 'dobj' and child.pos_ in ['NOUN', 'PROPN']:
                    phrase = phrase + ' ' + child.text
                    dobj_found = True
                    break
            
            if not dobj_found:
                for token in doc:
                    if token.dep_ == 'dobj' and token.pos_ in ['NOUN', 'PROPN']:
                        if token.head.lemma_.lower() not in ['tell', 'know', 'show', 'ask']:
                            phrase = phrase + ' ' + token.text
                            break
            
            return phrase
    
    # 4. Составные существительные
    for t in reversed(doc):
        if t.dep_ == 'compound' and t.pos_ == 'NOUN':
            parent = t.head
            if parent.pos_ == 'NOUN':
                adjectives = []
                current = parent
                while current.pos_ in ['NOUN', 'PROPN']:
                    for child in current.children:
                        if child.dep_ == 'amod' and child.pos_ == 'ADJ':
                            adjectives.append(child.text)
                    if current.head == current:
                        break
                    current = current.head
                
                parts = []
                if adjectives:
                    parts.extend(adjectives)
                parts.append(t.text)
                parts.append(parent.text)
                
                top_noun = parent
                while top_noun.head.pos_ in ['NOUN', 'PROPN'] and top_noun.head != top_noun:
                    top_noun = top_noun.head
                
                verb_candidate = top_noun.head
                if verb_candidate.pos_ in ['VERB', 'NOUN']:
                    parts.append(verb_candidate.lemma_)
                
                return ' '.join(parts)

    # 5. Прямое дополнение
    for t in reversed(doc):
        if t.dep_ == 'dobj' and t.pos_ in ['NOUN', 'PROPN', 'VERB']:
            verb = t.head

            if verb.lemma_.lower() in ['tell', 'show', 'explain', 'ask']:
                for token in doc:
                    if token.pos_ == 'VERB' and token.lemma_.lower() not in ['tell', 'show', 'explain', 'ask']:
                        verb = token
                        break
            
            verb_lemma = verb.lemma_ if hasattr(verb, 'lemma_') else verb.text
            return verb_lemma + 'ing' + ' ' + t.text
    
    return False


# Поиск информации в Wikipedia
def wiki(concept, depth=0):
    MAX_DEPTH = 2
    
    if depth >= MAX_DEPTH:
        return f"Could not find information after {MAX_DEPTH} search attempts: '{concept}'"
    
    print(f" Ищем: '{concept}' (глубина: {depth})")
    
    try:
        print(f"  Поиск статей для '{concept}'...")
        search_results = wikipedia.search(concept, results=5)
        
        if not search_results:
            return f"Nothing found for query: '{concept}'"
        
        print(f"     Найдено: {search_results}")
        
        best_match = None
        concept_lower = concept.lower()
        
        for result in search_results:
            result_lower = result.lower()

            if len(concept.split()) == 1:
                if '(' in result and ')' in result:
                    content = result[result.find("(")+1:result.find(")")].lower()
                    if content not in ['animal', 'bird', 'mammal', 'plant', 'fruit']:
                        print(f"     Пропускаем узкую статью: '{result}'")
                        continue
            
            if concept_lower == result_lower:
                best_match = result
                print(f"     Точное совпадение: '{best_match}'")
                break
            elif concept_lower in result_lower:
                best_match = result
                print(f"    ≈ Частичное совпадение: '{best_match}'")
        
        if not best_match:
            best_match = search_results[0]
            print(f"    Берём первый результат: '{best_match}'")
        
        print(f"  Загружаем страницу: '{best_match}'...")
        wiki_resp = wikipedia.page(best_match, auto_suggest=False)
        print(f"    Заголовок: '{wiki_resp.title}'")
        print(f"    URL: {wiki_resp.url}")
        
        content = wiki_resp.content[:5000]
        doc = nlp(content)
        concept_words = concept_lower.split()
        concept_doc = nlp(concept)
        concept_lemmas = [token.lemma_.lower() for token in concept_doc]
        
        if len(concept_words) == 1:
            word = concept_words[0]
            word_lemma = concept_lemmas[0] if concept_lemmas else word
            
            scored_sentences = []
            
            for i, sent in enumerate(doc.sents):
                sent_text = sent.text
                sent_lower = sent_text.lower()
                sent_doc = nlp(sent_text)
                
                if word in sent_lower or word_lemma in sent_lower:
                    score = 0
                    
                    if sent.start < 1000:
                        score += 10
                    
                    for token in sent_doc:
                        token_text = token.text.lower()
                        token_lemma = token.lemma_.lower()
                        
                        if token_text == word or token_lemma == word_lemma:
                            if token.dep_ in ['nsubj', 'dobj', 'pobj', 'attr', 'root']:
                                score += 15
                            elif token.dep_ in ['amod', 'compound', 'nmod']:
                                score += 8
                            else:
                                score += 3
                    
                    if sent_lower.startswith(word + ' is') or \
                       sent_lower.startswith(word + ' are') or \
                       sent_lower.startswith('the ' + word + ' is') or \
                       sent_lower.startswith('a ' + word + ' is') or \
                       ' is a ' + word in sent_lower or \
                       ' are ' + word in sent_lower or \
                       word + ' refers to' in sent_lower or \
                       word + ' is the' in sent_lower:
                        score += 25
                    
                    for token in sent_doc:
                        if token.lemma_ in ['be', 'mean', 'refer', 'define']:
                            score += 10
                    
                    sent_len = len(sent_text.split())
                    if 5 <= sent_len <= 30:
                        score += 5
                    
                    scored_sentences.append((sent_text, score))
            
            if scored_sentences:
                scored_sentences.sort(key=lambda x: x[1], reverse=True)
                best_sent, best_score = scored_sentences[0]
                
                print(f"    Нашли {len(scored_sentences)} предложений, лучшее (score: {best_score})")
                
                if best_score >= 20:
                    return best_sent
                else:
                    print(f"    Лучший результат слабый, пробуем summary")
                    try:
                        summary = wikipedia.summary(best_match, sentences=2)
                        if len(summary) > 50:
                            return summary
                    except:
                        pass
                    
                    return best_sent
        
        else:
            scored_sentences = []
            
            for sent in doc.sents:
                sent_text = sent.text
                sent_lower = sent_text.lower()
                sent_doc = nlp(sent_text)
                
                words_found = sum(1 for word in concept_words if word in sent_lower)
                lemmas_found = sum(1 for lemma in concept_lemmas 
                                 if any(token.lemma_.lower() == lemma for token in sent_doc))
                
                score = words_found * 10 + lemmas_found * 8
                
                if words_found == len(concept_words):
                    score += 30
                
                if lemmas_found == len(concept_lemmas):
                    score += 25
                
                if len(concept_lemmas) >= 2:
                    for i, token in enumerate(sent_doc):
                        token_lemma = token.lemma_.lower()
                        
                        if token_lemma in concept_lemmas:
                            if token.dep_ == 'nsubj' and token.head.lemma_.lower() in concept_lemmas:
                                score += 20
                            
                            if token.dep_ == 'dobj' and token.head.lemma_.lower() in concept_lemmas:
                                score += 15
                
                if sent.start < 1500:
                    score += 8
                
                important_words = ['is', 'are', 'was', 'were', 'means', 'refers', 'defined']
                if any(word in sent_lower for word in important_words):
                    score += 5
                
                if '(' in sent_text and ')' in sent_text:
                    score -= 5
                
                if len(sent_text.split()) > 40:
                    score -= 3
                
                if score > 0:
                    scored_sentences.append((sent_text, score))
            
            if scored_sentences:
                scored_sentences.sort(key=lambda x: x[1], reverse=True)
                best_sent, best_score = scored_sentences[0]
                
                print(f"    Нашли {len(scored_sentences)} предложений, лучшее (score: {best_score})")
                
                if best_score >= 30:
                    return best_sent
                elif best_score >= 15:
                    if len(scored_sentences) > 1:
                        second_sent = scored_sentences[1][0]
                        if best_sent[:50] != second_sent[:50]:
                            return best_sent + " " + second_sent
                    
                    return best_sent
        
        print(f"    Не нашли подходящих предложений, используем summary")
        try:
            summary = wikipedia.summary(best_match, sentences=2)
            return summary
        except:
            first_sent = list(doc.sents)[0].text if list(doc.sents) else "No content found"
            return first_sent
    
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"    Неоднозначность для '{concept}'")
        options = e.options
        
        concept_lower = concept.lower()
        best_option = None
        best_score = 0
        
        for option in options:
            option_lower = option.lower()
            score = 0
            
            if option_lower == concept_lower:
                best_option = option
                print(f"    Нашли точное совпадение: '{option}'")
                break
            
            if option_lower.startswith(concept_lower + " "):
                score += 10
                if "(" in option and ")" in option:
                    score += 5
            
            if concept_lower in option_lower:
                score += 3
            
            word_count = len(option.split())
            if word_count <= 2:
                score += 2
            elif word_count > 4:
                score -= 1
            
            if "(" in option and ")" in option:
                content = option[option.find("(")+1:option.find(")")].lower()
                if content in ['fruit', 'food', 'animal', 'bird', 'plant', 'city', 'country', 'river', 'mountain']:
                    score += 8
                elif any(x in content for x in ['album', 'song', 'film', 'movie', 'book']):
                    score -= 5
            
            if score > best_score:
                best_score = score
                best_option = option
        
        if best_option and best_score >= 5:
            print(f"    Автоматически выбираем '{best_option}' (score: {best_score})")
            try:
                return wiki(best_option, depth + 1)
            except:
                pass
        
        filtered_options = []
        for option in options[:10]:
            option_lower = option.lower()
            
            if len(option.split()) > 5:
                continue
            
            if concept_lower not in option_lower:
                continue
            
            skip_keywords = ['disambiguation', 'list of', 'index of', 'category:', 'template:']
            if any(keyword in option_lower for keyword in skip_keywords):
                continue
            
            filtered_options.append(option)
            if len(filtered_options) >= 5:
                break
        
        if filtered_options:
            options_text = "\n".join(f"• {opt}" for opt in filtered_options)
            return f"Please clarify what you mean by '{concept}':\n{options_text}"
        
        options_text = "\n".join(f"• {opt}" for opt in options[:3])
        return f"Please clarify your query '{concept}'. Possible options:\n{options_text}"
    
    except Exception as e:
        print(f"    Ошибка при поиске: {e}")
        return f"Error searching for '{concept}': {str(e)}"


# Обработка изображений и определение пиццы
def photo_tags(filename):
    try:
        print(f"Анализируем изображение: {filename}")
        
        if not os.path.exists(filename):
            print("Файл не найден")
            return "image_file_not_found", False
        
        general_model = Model(
            user_id='clarifai',
            app_id='main', 
            model_id='general-image-recognition',
            pat=CLARIFAI_API_KEY
        )
        
        result = general_model.predict_by_filepath(filename, input_type="image")
        concepts = result.outputs[0].data.concepts
        
        # Проверяем, является ли это пиццей
        is_pizza = False
        for concept in concepts[:10]:
            if 'pizza' in concept.name.lower() and concept.value > 0.3:
                is_pizza = True
                print(f"   Обнаружена пицца! ({concept.name}: {concept.value:.2f})")
                break
        
        # Проверка категорий
        for concept in concepts[:10]:
            # Еда
            if concept.name == 'food' and concept.value > 0.3:
                try:
                    food_model = Model(
                        user_id='clarifai',
                        app_id='main', 
                        model_id='food-item-recognition',
                        pat=CLARIFAI_API_KEY
                    )
                    food_result = food_model.predict_by_filepath(filename, input_type="image")
                    food_concepts = food_result.outputs[0].data.concepts
                    
                    if food_concepts:
                        best_tag = food_concepts[0].name
                        # Проверяем, является ли это пиццей
                        if 'pizza' in best_tag.lower():
                            is_pizza = True
                            print(f"   Пицца обнаружена через food-модель: {best_tag}")
                        return best_tag, is_pizza
                except Exception as e:
                    print(f"   Не удалось уточнить через food-модель: {e}")

            # Одежда
            elif concept.name in ['apparel', 'clothing', 'fashion'] and concept.value > 0.3:
                print(f"   Обнаружена одежда ({concept.name}: {concept.value:.2f})! Используем apparel-модель...")
                try:
                    apparel_model = Model(
                        user_id='clarifai',
                        app_id='main', 
                        model_id='apparel-detection',
                        pat=CLARIFAI_API_KEY
                    )
                    apparel_result = apparel_model.predict_by_filepath(filename, input_type="image")
                    apparel_concepts = apparel_result.outputs[0].data.concepts
                    
                    if apparel_concepts:
                        best_tag = apparel_concepts[0].name
                        print(f"   Уточненный тег от apparel-модели: {best_tag}")
                        return best_tag, False
                except Exception as e:
                    print(f"   Не удалось уточнить через apparel-модель: {e}")
        
        # Лучший тег из общей модели
        if concepts:
            best_tag = concepts[0].name
            return best_tag, is_pizza
        else:
            return "object", False
            
    except Exception as e:
        print(f"Ошибка при анализе изображения: {e}")
        import traceback
        traceback.print_exc()
        return "object", False


# Проверка, упоминается ли пицца в тексте
def is_pizza_mentioned(text):
    doc = nlp_sm(text.lower())
    pizza_keywords = ['pizza', 'pie', 'order pizza', 'pizza menu', 'want pizza', 'buy pizza']
    
    text_lower = text.lower()
    for keyword in pizza_keywords:
        if keyword in text_lower:
            return True
    
    # Проверяем через NLP структуру
    for token in doc:
        if token.text in ['pizza', 'pie']:
            return True
        if token.dep_ == 'dobj' and token.text in ['pizza', 'pie']:
            return True
    
    return False


# ==================== ФУНКЦИИ ИЗ PIZZA_BOT ====================

def extract_intent(doc):
    verb = None
    dobj = None

    for token in doc:
        if token.dep_ == 'dobj':
            verb = token.head.text.lower()
            dobj = token.text.lower()
            break

    if not verb or not dobj:
        return None

    verb_list = [
        ('order', 'want', 'give', 'make', 'get', 'buy'), 
        ('show', 'find', 'see', 'look', 'display', 'list', 'need')
    ]

    dobj_list = [
        ('pizza', 'pie', 'dish', 'food', 'item'),
        ('menu', 'list', 'catalog', 'selection', 'offer')
    ]

    verb_syns = [item for item in verb_list if verb in item]
    dobj_syns = [item for item in dobj_list if dobj in item]

    if not verb_syns or not dobj_syns:
        return None
    
    if dobj in dobj_list[0]:
        intent = verb_syns[0][0] + dobj_syns[0][0].capitalize()
    else:
        intent = 'showPizza'

    return intent


def details_to_str(user_data):
    details = list()
    for key, value in user_data.items():
        details.append('{} - {}'.format(key, value))
    return "\n" + "\n".join(details) + "\n"


def extract_pizza_type_from_message(text):
    doc = nlp_sm(text.lower())
    
    pizza_types = ['margarita', 'pepperoni', 'hawaiian', 'vegetarian', 'cheese', 'meat']
    
    for token in doc:
        if token.text in pizza_types:
            return token.text.capitalize()
    
    for token in doc:
        if token.text in ['pizza', 'pie', 'dish']:
            for child in token.lefts:
                if child.dep_ in ('compound', 'amod'):
                    return child.text.capitalize()
    
    return None


# ==================== ОБРАБОТЧИКИ СООБЩЕНИЙ ====================

def start(update, context):
    welcome_text = (
        "Hi! I'm a bot who can find any information for you :)\n"
        "Just write a question or send a photo.\n"
        "If you want to order pizza, just say so or send a pizza photo!"
    )
    update.message.reply_text(welcome_text)
    return 'ORDERING'


def intent_ext(update, context):
    msg = update.message.text
    doc = nlp_sm(msg)
    
    # Если пользователь ответил "yes" после показа меню или фото пиццы
    if msg.lower() in ['yes', 'y', 'yeah', 'ok'] and context.user_data.get('pizza_mode', False):
        return show_menu(update, context)
    
    # Проверяем, упоминается ли пицца
    if is_pizza_mentioned(msg) or context.user_data.get('pizza_mode', False):
        # Запускаем pizza_bot логику
        context.user_data['pizza_mode'] = True
        
        # Проверяем намерение через NLP
        intent = extract_intent(doc)
        
        if intent == 'orderPizza':
            context.user_data['product'] = 'pizza'
            
            pizza_type = extract_pizza_type_from_message(msg)

            if pizza_type:
                context.user_data['type'] = pizza_type
                user_data = context.user_data
                update.message.reply_text(
                    "Your order has been placed.\n{}\nHave a nice day!".format(
                        details_to_str(user_data)
                    )
                )
                context.user_data['pizza_mode'] = False
                return ConversationHandler.END
            
            else:
                update.message.reply_text(
                    "We need some more information to place your order. "
                    "What type of pizza do you want?"
                )
                return 'ADD_INFO'
        
        elif intent == 'showPizza' or msg.lower() in ['menu', 'show menu', 'see menu']:
            return show_menu(update, context)
        
        # Если просто упоминание пиццы без четкого намерения
        else:
            update.message.reply_text(
                "Would you like to order a pizza or see our menu?\n"
                "Say 'order pizza' or 'show menu'"
            )
            return 'ORDERING'
    
    else:
        # Не пицца - используем wiki_bot логику
        doc_wiki = nlp(msg)
        concept = keyphrase(doc_wiki)
        
        if concept != False:
            print(f"Извлеченная ключевая фраза: '{concept}'")
            response = wiki(concept)
            
            if len(response) > 4000:
                response = response[:4000] + "...\n\n(сообщение сокращено)"
            update.message.reply_text(response)
        else:
            print(" Не удалось извлечь ключевую фразу")
            update.message.reply_text(
                "I can't understand your question. Try to reformulate.\n\n"
                "For example:\n"
                "• Information about Moscow'\n"
                "• How do horses sleep?"
            )
        
        return ConversationHandler.END


def add_info(update, context):
    msg = update.message.text
    doc = nlp_sm(msg)

    found_pizza_word = False

    for token in doc:
        if token.text.lower() in ['pizza', 'pie', 'dish']:
            found_pizza_word = True

            for child in token.lefts:
                if child.dep_ in ('compound', 'amod'):
                    context.user_data['type'] = child.text
                    break
            if 'type' not in context.user_data:
                for child in token.rights:
                    if child.dep_ in ('compound', 'amod'):
                        context.user_data['type'] = child.text
                        break

    if not found_pizza_word:
        for token in doc:
            if token.pos_ in ('ADJ', 'PROPN', 'NOUN'):
                context.user_data['type'] = token.text
                break

    if 'type' in context.user_data:
        user_data = context.user_data
        update.message.reply_text(
            "Your order has been placed.\n{}\nHave a nice day!".format(
                details_to_str(user_data)
            )
        )
        context.user_data['pizza_mode'] = False
        return ConversationHandler.END
    else:
        update.message.reply_text(
            "I didn't catch the pizza type. "
            "Please say something like: 'Margarita pizza', 'Pepperoni pizza', or just 'Pepperoni'."
        )
        return 'ADD_INFO'


def show_menu(update, context):
    menu_text = """
    🍕 Our Pizza Menu 🍕
    1. Margarita pizza $10
    2. Veg cheese pizza $11.99
    3. Sweet corn pizza $12.49
    4. Baby corn pizza $12.49
    5. Onion & Capsicum Pizza $11.99
    6. Tomato pizza $12.49
    7. Double Cheese Pizza $17.99
    8. Italian Pizza $18.99
    9. Mexican Pizza $19.49
    10. Spanish Pizza $19.49
    11. Mushroom Pizza $16.99
    """
    update.message.reply_text(menu_text)
    
    update.message.reply_text(
        "Would you like to order something from the menu?\n"
        "Say 'yes' to order or 'no' to finish."
    )
    return 'WAITING_ORDER_CONFIRMATION'


def handle_order_confirmation(update, context):
    user_response = update.message.text.lower()
    
    if user_response in ['yes', 'y', 'yeah', 'ok', 'order']:
        update.message.reply_text(
            "Great! To order, please say: 'I want a [pizza name] pizza'\n"
            "For example: 'I want a Margarita pizza'"
        )
        context.user_data['pizza_mode'] = True
        return 'ORDERING' 
    
    elif user_response in ['no', 'n', 'nah', 'exit']:
        update.message.reply_text("Okay! Have a nice day!")
        context.user_data['pizza_mode'] = False
        return ConversationHandler.END
    
    else:
        update.message.reply_text("Please answer 'yes' or 'no'.")
        return 'WAITING_ORDER_CONFIRMATION'


def cancel(update, context):
    update.message.reply_text("Have a nice day!")
    context.user_data['pizza_mode'] = False
    return ConversationHandler.END




# Обработчик фотографий
def photo(update, context):
    user = update.message.from_user
    
    try:
        # Скачиваем фото
        photo_file = update.message.photo[-1].get_file()
        filename = f"temp_{photo_file.file_id}.jpg"
        photo_file.download(filename)
        
        # Анализируем фото через Clarifai
        concept, is_pizza = photo_tags(filename)
        print(f"   Получен тег: {concept}, является пиццей: {is_pizza}")
        
        if concept != "image_file_not_found":
            if is_pizza:
                # Если это пицца - сначала показываем информацию из Wikipedia
                print("   Это пицца! Показываем информацию из Wikipedia...")
                update.message.reply_text(f"Analyzing the image... I can see: {concept}")
                response = wiki(concept)
                
                if len(response) > 4000:
                    response = response[:4000] + "...\n\n(сообщение сокращено)"
                update.message.reply_text(response)
                
                # Затем запускаем pizza_bot
                print("   Запускаем pizza_bot...")
                update.message.reply_text(
                    "\n\nWould you like to order a pizza? "
                    "Say 'yes' to see the menu or describe what pizza you want!"
                )
                # Устанавливаем флаг, что пользователь в режиме заказа пиццы
                context.user_data['pizza_mode'] = True
                # Возвращаем состояние ORDERING для обработки следующего сообщения через pizza_bot
                return 'ORDERING'
            else:
                # Если не пицца - используем обычную wiki логику
                update.message.reply_text(f"Analyzing the image... I can see: {concept}")
                response = wiki(concept)
                
                if len(response) > 4000:
                    response = response[:4000] + "...\n\n(сообщение сокращено)"
                update.message.reply_text(response)
                # Завершаем разговор, так как это не пицца
                return ConversationHandler.END
        
        # Удаляем временный файл
        if os.path.exists(filename):
            os.remove(filename)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        update.message.reply_text(f"Photo processing error: {str(e)}")
        return ConversationHandler.END
    
    # Если не пицца, завершаем разговор
    return ConversationHandler.END


# Обработчик ошибок
def error(update, context):
    print(f"Ошибка: {context.error}")
    if update and update.message:
        update.message.reply_text("Произошла ошибка при обработке запроса")


def main():
    print("Запуск объединенного бота...")
    
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    
    # ConversationHandler обрабатывает все сообщения
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start),
            MessageHandler(Filters.text & ~Filters.command, intent_ext),
            MessageHandler(Filters.photo, photo)
        ], 
        states={ 
            'ORDERING': [
                MessageHandler(Filters.text & ~Filters.command, intent_ext),
                MessageHandler(Filters.photo, photo)
            ],
            'ADD_INFO': [MessageHandler(Filters.text & ~Filters.command, add_info)],
            'SHOW_MENU': [MessageHandler(Filters.text & ~Filters.command, show_menu)],
            'WAITING_ORDER_CONFIRMATION': [MessageHandler(Filters.text & ~Filters.command, handle_order_confirmation)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
        allow_reentry=True
    )
    
    dp.add_handler(conv_handler)
    dp.add_error_handler(error)
    
    print("Бот запущен. Ожидаем сообщения...")
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()



