import json
import pandas as pd
import numpy as np
import spacy
import inflect
from string import punctuation
import random
import re

SINGULAR = "person"
PLURAL = "they"

class SNLI_PARSER():
    def __init__(self, nlp, pronoun_dict):
        self.nlp = nlp
        self.inflect = inflect.engine()
        self.pronoun_dict = pronoun_dict

    def is_singular(self, noun):
        # inflect.signular_noun returns False if the word is singular
        return not self.inflect.singular_noun(noun) 

    def get_pronoun_to_replace(self, sent2):
        try:
            doc = self.nlp(sent2)
        except:
            return None
        
        # identify the subject of the sentence
        sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj")]
        
        if len(sub_toks) != 1:
            # TODO - what to do when we have no nsubj, or more than one?
            return None
        
        # Get the modifiers of the subject
        children, children_ind = [], []
        for child in sub_toks[0].children:
            children.append(child.text)
            children_ind.append(child.i)
            
        subject = sub_toks[0].text
        subject_ind = sub_toks[0].i
        
        # Get replacing pronouns
        if self.is_singular(subject):
            pronoun = self.pronoun_dict.get(subject.lower(), None)
            if pronoun == "person":
                if random.random() > 0.5:
                    pronoun = "she"
                else:
                    pronoun = "he"

        else:
            pronoun = PLURAL

        if pronoun is None:
            return None
            
        min_token_ind = min(children_ind + [subject_ind]) 
        
        prev = [tok.text for tok in doc[:min_token_ind]]
        after = [tok.text for tok in doc[subject_ind + 1:]]

        return (prev, after, pronoun)
        
    def get_sent_from_tokens(self, tokens):
        new_sent = ""
        for token in tokens:
            if token in punctuation:
                new_sent = new_sent[:-1]
            new_sent += token + " "

        new_sent = new_sent.strip()
        new_sent = new_sent[0].upper() + new_sent[1:]

        return new_sent

    def augment_false_example(self, sent1, sent2, pronoun, prev, after):
        '''
        Generates false examples for given sentences
        We find subjects/objects in original sentence that match the following criteria
            1. Plurality matches that of the pronoun we are replacing (hence the pluarilty of subject of sent2)
            2. The word does not appear in sent2, as then the sentence probably doesn't make much sense
        '''
        try:
            doc = self.nlp(sent1)
        except:
            return [] 
        
        # identify the subject and objects of the sentence
        toks = [tok for tok in doc if (tok.dep_ in ["subj", "dobj", "pobj"])]
        
        candidates = []
        for tok in toks:
            word = tok.text
             
            is_singular = self.is_singular(word)
            
            # if plurality does not match that of our subject of interest
            if (not is_singular and pronoun != PLURAL) or (is_singular and pronoun == PLURAL): 
                continue
               
            # check if the word appears in sent 2
            if word.lower() in sent2:
                continue

            # We can use it as candidate only if they share same pronoun
            candidate_pronoun = self.pronoun_dict.get(word.lower(), None)
            if candidate_pronoun is None:
                continue

            if candidate_pronoun == SINGULAR and pronoun in ['he', 'she']:
                candidates.append(word)
            elif candidate_pronoun == pronoun:
                candidates.append(word)    
            else:
                continue

        # Generate new sentences by replacing with a word from sentence 1
        false_examples = [self.get_sent_from_tokens(prev + [cand] + after) for cand in candidates]
        
        return false_examples    

    def clean_sentence(self, sent):
        # remove random spaces
        return re.sub(' +', ' ', sent)

    def augment(self, sent1, sent2):
        sent1 = self.clean_sentence(sent1)
        sent2 = self.clean_sentence(sent2)

        pronoun_result = self.get_pronoun_to_replace(sent2)
        if pronoun_result is None:
            return None
        
        (prev, after, pronoun) = pronoun_result
        
        # Augment sentence
        new_sent1 = sent1 + " " + self.get_sent_from_tokens(prev + [pronoun] + after)
        
        # Original sentence 2 is an entailment example
        true_example = sent2.strip() 
        true_example = true_example[0].upper() + true_example[1:]
        
        # generate false entailment examples 
        false_examples = self.augment_false_example(sent1, sent2, pronoun, prev, after)
        
        return (new_sent1, true_example, false_examples)
