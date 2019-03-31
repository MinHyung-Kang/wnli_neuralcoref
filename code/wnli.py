from string import punctuation

ENTAILMENT = 1
NOT_ENTAILMENT = 0
MAJORITY = NOT_ENTAILMENT

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

class WNLI():
    '''
        If use_coref is False, we always return majority label
    '''
    def __init__(self, nlp, data, majority=MAJORITY, use_coref=True, debug=False):
        self.nlp = nlp
        self.data = data
        self.debug = debug
        self.use_coref = use_coref
        self.majority = majority
        self.none_count = None
        
    def remove_article(self, sen):
        sen = sen.replace("the ", "").replace("an ", "").replace("a ", "").strip()

        return sen
        
    '''
    Find a possible overlap between sentence1 and sentence 2
    Return (sen1_ref_ind, sen1_ref, sen2_ref_ind, sen2_ref, result_query1, result_query2)
    '''    
    def find_overlap(self, sen1, sen2):
        ori_sen1 = sen1
        ori_sen2_tokens = sen2.split(" ")

        sen1 = strip_punctuation(sen1.lower())
        sen2 = strip_punctuation(sen2.lower())
        tokens = sen2.split(" ")

        result = None
        # allow for up to 3 words
        for slack in range(5):
            for ind, token in enumerate(tokens[:len(tokens)-slack]):
                query1 = " ".join(tokens[:ind])
                query2 = " ".join(tokens[ind + slack+1:])

                if query1 in sen1 and query2 in sen1:
                    sen1_ref_result = self.find_sen1_ref(ori_sen1, query1.strip(), query2.strip())

                    # If we see that query2 is before query1, we assume that we parsed this wrong
                    if sen1_ref_result is None:
                        continue

                    (sen1_ref_ind, sen1_ref) = sen1_ref_result

                    result_query1 = " ".join(ori_sen2_tokens[:ind]).strip() 
                    result_query2 = " ".join(ori_sen2_tokens[ind + slack+1:]).strip()
                    sen2_ref = " ".join(ori_sen2_tokens[ind:ind + slack + 1])

                    return (sen1_ref_ind, sen1_ref, ind, sen2_ref, result_query1, result_query2)

        return None
    
    '''
    Find the corresponding reference in sentence 1 from sentence 2
    '''
    def find_sen1_ref(self, sen1, query1, query2):
        ori_sen1_tokens = sen1.split(" ")
        sen1_tokens = strip_punctuation(sen1.lower()).split(" ")

        query1_ind, query2_ind = -1, -1
        query1_len = 0 if len(query1) == 0 else len(query1.split(" "))
        query2_len = 0 if len(query2) == 0 else len(query2.split(" "))

        # Get the indices of each query
        for ind in range(len(sen1_tokens)):
            sen1_substring = " ".join(sen1_tokens[ind:])

            if query1 in sen1_substring:
                query1_ind = ind

            if query2 in sen1_substring:
                query2_ind = ind

            # we no longer have the query in the substring
            if query1_ind != ind and query2_ind != ind and query1_ind != -1 and query2_ind != -1:
                break
                
        # Sanity check: check that the words actually are all the same in tokens
        if query1_len > 0:
            sen1_substring = sen1_tokens[query1_ind:query1_ind + query1_len]
            if not all([x == y for x, y in zip(sen1_substring, query1.split(" "))]):
                return None
            
        if query2_len > 0:
            sen1_substring = sen1_tokens[query2_ind:query2_ind + query2_len]
            if not all([x == y for x, y in zip(sen1_substring, query2.split(" "))]):
                return None
            

        # Do a sanity check, making sure we have query 2 after query 1
        if query1_len > 1 and query2_len > 0 and query1_ind > query2_ind:
            return None

        # if query 1 is "the" we know the word should come before query 2 in original sentence
        # if query 2 has more than length 0 we are pretty positive that the query goes inbetween query 1 and query2
        if query1 in ["the","a","an"] or query2_len > 0:
            ori_ref_ind = query2_ind - 1
            ori_ref = ori_sen1_tokens[ori_ref_ind]

        # if query2 has length 0, we know that reference should be after query 1
        elif query2_len == 0:
            ori_ref_ind = query1_ind + query1_len

            if ori_sen1_tokens[ori_ref_ind] in ["the", "a", "an", "of"]:
                ori_ref_ind += 1

            ori_ref = ori_sen1_tokens[ori_ref_ind]
        else:
            if self.debug:
                raise ValueError("Failed to parse : " + sen1)
                
            return None

        return (ori_ref_ind, ori_ref)
    
    '''
    Using the coreference model, try to make prediction
    If the model fails to give us an answer, just return None
    '''
    def get_label(self, sen1, sen1_ref_ind, sen1_ref, sen2_ref):
        doc_sen1 = self.nlp(sen1)

        # we don't have any coreference detected, so just return None
        if doc_sen1._.coref_clusters is None:
            if self.debug:
                print("No coreference detected, returning None")
            return None

        # the indices of token by nlp does not match the token from previous parsing - hence, manually align (might incur some error)
        spacy_ind = sen1_ref_ind
        matched = False

        while not matched:
            if spacy_ind == len(doc_sen1):
                if self.debug:
                    print("Spacy_index went over the index of sentence")
                return None

            if strip_punctuation(str(doc_sen1[spacy_ind]).lower()) == strip_punctuation(sen1_ref.lower()):
                matched = True
            else:
                spacy_ind += 1

        token = doc_sen1[spacy_ind]

        # we think the token is not in coref - we can't do anything, return None
        if not token._.in_coref:
            if self.debug:
                print("Token not in coref, returning None")
            return None

        cluster = token._.coref_clusters[0]
        
        query =  self.remove_article(strip_punctuation(sen2_ref.lower()))
        
        if self.debug:
            print("Query:" + query)
            print("Cluster : " + str(cluster))

        if query in strip_punctuation(str(cluster.main).lower()):
            return ENTAILMENT

        for mention in cluster.mentions:
            if query in strip_punctuation(str(mention).lower()):
                return ENTAILMENT

        return NOT_ENTAILMENT

    def predict_single(self, ind, sen1, sen2):
        if not self.use_coref:
            return self.majority
        
        if self.debug:
            print("="*50)
        
        result = self.find_overlap(sen1, sen2)
        pred_label = None

        if result is not None:
            (sen1_ref_ind, sen1_ref, sen2_ref_ind, sen2_ref, result_query1, result_query2) = result       
            pred_label = self.get_label(sen1, sen1_ref_ind, sen1_ref, sen2_ref)
            
            if self.debug:
                print("[sen1]: " + sen1)
                print("[sen2]: " + sen2)
                print("[sen1_ref_ind]: " + str(sen1_ref_ind))
                print("[sen1_ref]: " + sen1_ref)
                print("[sen2_ref_ind]: " + str(sen2_ref_ind))
                print("[sen2_ref]: " + sen2_ref)
                print("[result_query1]: " + result_query1)
                print("[result_query2]: " + result_query2)
                print("[pred_label]:" + str(pred_label) + (" -> " + str(self.majority) if pred_label is None else ""))
                print("[actual_label]:" + str(self.data['label'][ind]))
                
        if pred_label is None:
            self.none_count += 1
            
        return self.majority if pred_label is None else pred_label

    def predict(self):
        self.none_count = 0
        labels = [self.predict_single(ind, row['sentence1'], row['sentence2']) for ind, row in self.data.iterrows()]
        
        if self.use_coref:
            print("Could not use coref model for {}/{} examples".format(self.none_count, len(self.data)))

        return labels

    def score(self, pred_labels):
        true_labels = self.data['label']
        return (pred_labels == true_labels).mean()