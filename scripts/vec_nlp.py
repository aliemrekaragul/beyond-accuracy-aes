from textstat import flesch_reading_ease
import math
from collections import OrderedDict
from spellchecker import SpellChecker
from statistics import mean
import en_core_web_sm
import pandas as pd
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

def detect_voice(doc):
    passive_count = sum(1 for token in doc if token.dep_ == 'auxpass' and any((child.dep_ == 'nsubjpass' or child.dep_ == 'csubjpass') for child in token.head.children))
    active_count = sum(1 for token in doc if (token.dep_ == 'nsubj' or token.dep_ == 'csubj') and token.head.pos_ == 'VERB')
    return passive_count, active_count

def max_depth(token, current_depth=0):
    if not list(token.children):
        return current_depth
    return max(max_depth(child, current_depth + 1) for child in token.children)

def extract_features(essay):    
    spell = SpellChecker()
    nlp = en_core_web_sm.load()
    doc = nlp(essay)   

    # WORD LEVEL FEATURES
    valid_token_list = list(OrderedDict((token.text, None) for token in doc if token.is_alpha).keys())
    word_count = len(valid_token_list)
    unique_lemma_count = len(set([token.lemma_ for token in doc if token.is_alpha]))
    if word_count == 0:
        return {}
    # named entities in ASAP essays are pre-tagged with @
    pre_tagged_named_entity_count = sum(1 for word in doc if word.text.startswith('@'))
    spacy_named_entity_count = len(doc.ents)

    misspelled_words = spell.unknown(valid_token_list)
    named_entity_ratio = (pre_tagged_named_entity_count + spacy_named_entity_count) / word_count
    spelling_error_ratio = len(misspelled_words)/word_count if word_count > 0 else 100
    lemma_ratio = unique_lemma_count / word_count
    preposition_ratio = (sum(1 for tok in doc if tok.dep_ == 'prep')) / word_count
    negation_ratio = (sum(1 for tok in doc if tok.dep_ == 'neg'))  / word_count
    subordinate_conjunction_ratio = (sum(1 for tok in doc if tok.pos_ == 'SCONJ')) / word_count
    coordinating_conjunction_ratio = (sum(1 for tok in doc if tok.pos_ == 'CCONJ')) / word_count
    correlative_conjunction_ratio = (sum(1 for tok in doc if tok.dep_ == 'preconj')) / word_count
    foreign_word_ratio = (sum(1 for tok in doc if tok.tag_ == 'FW')) / word_count
    hyphenated_word_ratio = (sum(1 for tok in doc if tok.tag_ == 'HYPH')) / word_count
    unknown_word_ratio = (sum(1 for tok in doc if tok.tag_ == 'XX')) / word_count


    word_level_features = {
        'word_count': word_count,
        'unique_lemma_count': unique_lemma_count,
        'named_entity_ratio': named_entity_ratio,
        'spelling_error_ratio': spelling_error_ratio,
        'lemma_ratio': lemma_ratio,
        'preposition_ratio': preposition_ratio,
        'negation_ratio': negation_ratio,
        'subordinate_conjunction_ratio': subordinate_conjunction_ratio,
        'coordinating_conjunction_ratio': coordinating_conjunction_ratio,
        'correlative_conjunction_ratio': correlative_conjunction_ratio,
        'foreign_word_ratio': foreign_word_ratio,
        'hyphenated_word_ratio': hyphenated_word_ratio,
        'unknown_word_ratio': unknown_word_ratio
    }
    # SENTENCE LEVEL FEATURES
    sentence_count = len(list(doc.sents))
    sentence_lengths = [len(sent) for sent in doc.sents]
    passive_count, active_count = detect_voice(doc)
    relative_clause_count = sum(1 for tok in doc if tok.dep_ == 'relcl')
    complement_clause_count = sum(1 for tok in doc if tok.dep_ == 'ccomp')
    adverbial_clause_count = sum(1 for tok in doc if tok.dep_ == 'advcl')
    infinitival_clause_count = sum(1 for tok in doc if tok.dep_ == 'acl')
    clausal_pass_subj_count = sum(1 for tok in doc if tok.dep_ == 'csubjpass')
    clausal_act_subj_count = sum(1 for tok in doc if tok.dep_ == 'csubj')   
    clause_count = relative_clause_count + complement_clause_count + adverbial_clause_count + infinitival_clause_count + clausal_pass_subj_count + clausal_act_subj_count    
    root_tokens = [token for token in doc if token.head == token]

    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    passive_ratio = passive_count / sentence_count if sentence_count > 0 else 0
    active_ratio = active_count / sentence_count if sentence_count > 0 else 0
    relative_clause_ratio = relative_clause_count / sentence_count if sentence_count > 0 else 0
    complement_clause_ratio = complement_clause_count / sentence_count if sentence_count > 0 else 0
    adverbial_clause_ratio = adverbial_clause_count / sentence_count if sentence_count > 0 else 0
    infinitival_clause_ratio = infinitival_clause_count / sentence_count if sentence_count > 0 else 0
    clausal_passive_ratio = clausal_pass_subj_count / clause_count if clause_count > 0 else 0
    clausal_active_ratio = clausal_act_subj_count / clause_count if clause_count > 0 else 0
    clause_ratio = clause_count / sentence_count if sentence_count > 0 else 0
    readability = flesch_reading_ease(essay)
    entropy_sentence_length = sum( [-length / word_count * math.log(length / word_count) for length in sentence_lengths if length > 0] )
    tree_depth = mean(max_depth(token) for token in root_tokens) if root_tokens else 0
    parataxis_count = sum(1 for tok in doc if tok.dep_ == 'parataxis')
    direct_object_count = sum(1 for tok in doc if tok.dep_ == 'dobj')
    nominal_subject_count = sum(1 for tok in doc if tok.dep_ == 'nsubj')
    adverbial_modifier_count = sum(1 for tok in doc if tok.dep_ == 'advmod')
    adjectival_modifier_count = sum(1 for tok in doc if tok.dep_ == 'amod')
    nominal_modifier_count = sum(1 for tok in doc if tok.dep_ == 'nmod')

    sentence_level_features = {
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'passive_ratio': passive_ratio,
        'active_ratio': active_ratio,
        'relative_clause_ratio': relative_clause_ratio,
        'complement_clause_ratio': complement_clause_ratio,
        'adverbial_clause_ratio': adverbial_clause_ratio,
        'infinitival_clause_ratio': infinitival_clause_ratio,
        'clause_ratio': clause_ratio,
        'clausal_passive_ratio': clausal_passive_ratio,
        'clausal_active_ratio': clausal_active_ratio,
        'readability': readability,
        'entropy_sentence_length': entropy_sentence_length,
        'tree_depth': tree_depth,
        'parataxis_count': parataxis_count,
        'direct_object_count': direct_object_count,
        'nominal_subject_count': nominal_subject_count,
        'adverbial_modifier_count': adverbial_modifier_count,
        'adjectival_modifier_count': adjectival_modifier_count,
        'nominal_modifier_count': nominal_modifier_count
    }
    # PUNCTUAL FEATURES
    punctual_count = sum(1 for token in doc if token.is_punct)
    punctuation_variety = len(set(token.text for token in doc if token.is_punct))
    average_punctuation_per_sentence = sum(1 for token in doc if token.is_punct) / sentence_count if sentence_count > 0 else 0
    superfluous_punctuation_ratio = sum(1 for token in doc if token.is_punct and token.tag_ == "NFP") / punctual_count if punctual_count > 0 else 0
    punctuation_features = {
        'punctuation_variety': punctuation_variety,
        'average_punctuation_per_sentence': average_punctuation_per_sentence,
        'superfluous_punctuation_ratio': superfluous_punctuation_ratio
    }
    # PoS FEATURES
    adjective_count = sum(1 for token in doc if token.pos_ == 'ADJ')
    adjective_word_ratio = adjective_count / word_count
    affixed_adjective_ratio = sum(1 for token in doc if token.tag_ == 'AFX') / adjective_count if adjective_count > 0 else 0
    base_adjective_ratio = sum(1 for token in doc if token.tag_ == 'JJ') / adjective_count if adjective_count > 0 else 0
    comparative_adjective_ratio = sum(1 for token in doc if token.tag_ == 'JJR') / adjective_count if adjective_count > 0 else 0
    superlative_adjective_ratio = sum(1 for token in doc if token.tag_ == 'JJS') / adjective_count if adjective_count > 0 else 0
    predeterminer_adj_ratio = sum(1 for token in doc if token.tag_ == 'PDT') / adjective_count if adjective_count > 0 else 0
    quantifier_adj_ratio = sum(1 for token in doc if token.dep_ == 'quantmod') / adjective_count if adjective_count > 0 else 0
    possessive_adj_ratio = sum(1 for token in doc if token.tag_ == 'PRP$') / adjective_count if adjective_count > 0 else 0
    wh_adj_ratio = sum(1 for token in doc if (token.tag_ == 'WP$' or token.tag_ == 'WDT')) / adjective_count if adjective_count > 0 else 0
    #adverbs
    adverb_count = sum(1 for token in doc if token.pos_ == 'ADV')
    adverb_word_ratio = adverb_count / word_count
    base_adverb_ratio = sum(1 for token in doc if token.tag_ == 'RB') / adverb_count if adverb_count > 0 else 0
    comparative_adverb_ratio = sum(1 for token in doc if token.tag_ == 'RBR') / adverb_count if adverb_count > 0 else 0
    superlative_adverb_ratio = sum(1 for token in doc if token.tag_ == 'RBS') / adverb_count if adverb_count > 0 else 0
    wh_adverb_ratio = sum(1 for token in doc if token.tag_ == 'WRB') / adverb_count if adverb_count > 0 else 0
    existential_there_ratio = sum(1 for token in doc if token.tag_ == 'EX') / adverb_count if adverb_count > 0 else 0
    particle_adverb_ratio = sum(1 for token in doc if token.tag_ == 'RP') / adverb_count if adverb_count > 0 else 0
    #nouns
    noun_count = sum(1 for token in doc if token.pos_ == 'NOUN')
    noun_word_ratio = noun_count / word_count
    singular_noun_ratio = sum(1 for token in doc if token.tag_ == 'NN') / noun_count if noun_count > 0 else 0
    plural_noun_ratio = sum(1 for token in doc if token.tag_ == 'NNS') / noun_count if noun_count > 0 else 0
    proper_noun_ratio = sum(1 for token in doc if token.tag_ == 'NNP') / noun_count if noun_count > 0 else 0
    #verbs
    verb_count = sum(1 for token in doc if token.pos_ == 'VERB')
    verb_word_ratio = verb_count / word_count
    base_form_ratio = sum(1 for token in doc if token.tag_ == 'VB') / verb_count if verb_count > 0 else 0
    past_form_ratio = sum(1 for token in doc if token.tag_ == 'VBD') / verb_count if verb_count > 0 else 0
    present_form_ratio = sum(1 for token in doc if (token.tag_ == 'VBP' or token.tag_ == 'VBZ') ) / verb_count if verb_count > 0 else 0
    gerund_form_ratio = sum(1 for token in doc if token.tag_ == 'VBG') / verb_count if verb_count > 0 else 0
    to_infinitive_form_ratio = sum(1 for token in doc if token.tag_ == 'TO') / verb_count if verb_count > 0 else 0
    past_participle_form_ratio = sum(1 for token in doc if token.tag_ == 'VBN') / verb_count if verb_count > 0 else 0
    modal_verb_ratio = sum(1 for token in doc if token.tag_ == 'MD') / verb_count if verb_count > 0 else 0
    auxiliary_verb_ratio = sum(1 for token in doc if token.pos_ == 'AUX') / verb_count if verb_count > 0 else 0
    phrasal_verb_ratio = sum(1 for token in doc if token.dep_ == 'prt') / verb_count if verb_count > 0 else 0
    #pronouns
    personal_pronoun_ratio = sum(1 for token in doc if token.tag_ == 'PRP') / word_count
    wh_pronoun_count = sum(1 for token in doc if token.tag_ == 'WP')
    #others
    interjection_count = sum(1 for token in doc if token.pos_ == 'INTJ')
    symbol_count = sum(1 for token in doc if token.pos_ == 'SYM')
    numeral_count = sum(1 for token in doc if token.pos_ == 'NUM')

    pos_features = {
        'adjective_word_ratio': adjective_word_ratio,
        'affixed_adjective_ratio': affixed_adjective_ratio,
        'base_adjective_ratio': base_adjective_ratio,
        'comparative_adjective_ratio': comparative_adjective_ratio,
        'superlative_adjective_ratio': superlative_adjective_ratio,
        'predeterminer_adj_ratio': predeterminer_adj_ratio,
        'quantifier_adj_ratio': quantifier_adj_ratio,
        'possessive_adj_ratio': possessive_adj_ratio,
        'wh_adj_ratio': wh_adj_ratio,
        'adverb_word_ratio': adverb_word_ratio,
        'base_adverb_ratio': base_adverb_ratio,
        'comparative_adverb_ratio': comparative_adverb_ratio,
        'superlative_adverb_ratio': superlative_adverb_ratio,
        'wh_adverb_ratio': wh_adverb_ratio,
        'existential_there_ratio': existential_there_ratio,
        'particle_adverb_ratio': particle_adverb_ratio,
        'noun_word_ratio': noun_word_ratio,
        'singular_noun_ratio': singular_noun_ratio,
        'plural_noun_ratio': plural_noun_ratio,
        'proper_noun_ratio': proper_noun_ratio,
        'verb_word_ratio': verb_word_ratio,
        'base_form_ratio': base_form_ratio,
        'past_form_ratio': past_form_ratio,
        'present_form_ratio': present_form_ratio,
        'gerund_form_ratio': gerund_form_ratio,
        'to_infinitive_form_ratio': to_infinitive_form_ratio,
        'past_participle_form_ratio': past_participle_form_ratio,
        'modal_verb_ratio': modal_verb_ratio,
        'auxiliary_verb_ratio': auxiliary_verb_ratio,
        'phrasal_verb_ratio': phrasal_verb_ratio,
        'personal_pronoun_ratio': personal_pronoun_ratio,
        'wh_pronoun_count': wh_pronoun_count,
        'interjection_count': interjection_count,
        'symbol_count': symbol_count,
        'numeral_count': numeral_count
    }

    feature_set = { }
    feature_set.update(word_level_features)
    feature_set.update(sentence_level_features)
    feature_set.update(punctuation_features)
    feature_set.update(pos_features)
    
    return feature_set

import argparse

def run(input_path, output_path=None, model_dir=None):
    if output_path is None:
        output_path = os.path.dirname(input_path)
    if model_dir is None:
        model_dir = "models"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    id_col_name = "essay_id"
    essay_col_name = "essay"
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, encoding='ISO-8859-1')
    essays = df[essay_col_name]
    essay_ids = df[id_col_name]
    
    tqdm.pandas()
    print("Extracting hand-crafted features...")
    features = essays.progress_apply(extract_features)
    features_df = pd.json_normalize(features)
    
    scaler = MinMaxScaler()
    features_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)

    joblib.dump(scaler, os.path.join(model_dir, 'hand_craft_scaler.pkl'))
    features_df.insert(0, id_col_name, essay_ids)
    
    output_file = os.path.join(output_path, "hand-crafted-features.csv")
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand-crafted NLP features from essays.")
    parser.add_argument("--input", default=r"data\asap-set-7-essays.csv", help="Path to the input CSV file.")
    parser.add_argument("--output", help="Directory to save the output CSV. Defaults to input directory.")
    parser.add_argument("--model_dir", default="models", help="Directory to save the scaler model.")
    
    args = parser.parse_args()
    run(args.input, args.output, args.model_dir)