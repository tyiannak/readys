import Levenshtein
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner


def to_translation_map(iterable): 
    #a function to make a set of junk (None) characters
    return {key: None for key in iterable}


def text_to_text_alignment_and_score(text_ref, text_pred):
    """
    Find a word to word alignment between two texts, considering the first is 
    the reference and the second the predicted
    :param text_ref: text reference
    :param text_pred: predicted text
    :return: 
    """

    text_ref = text_ref.lower()
    text_pred = text_pred.lower()
    iterable = [".", ","]
    # convert the reference text in order not to contain , and (junk characters)
    translation_map = str.maketrans(to_translation_map(iterable))
    text_ref = text_ref.translate(translation_map)

    # Create sequences to be aligned.
    a = Sequence(text_ref.split())
    b = Sequence(text_pred.split())
    
    # Create a vocabulary and encode the sequences.
    v = Vocabulary()
    a_enc = v.encodeSequence(a)
    b_enc = v.encodeSequence(b)

    # Create a scoring and align the sequences using global aligner.
    scoring = SimpleScoring(1, 0)
    aligner = GlobalSequenceAligner(scoring, 0)
    f, score, encodeds = aligner.align(a_enc, b_enc, text_ref.split(),
                                       text_pred.split(), backtrace=True)

    # get the first allignment if exists:
    if len(encodeds) > 0:
        alignment = v.decodeSequenceAlignment(encodeds[0])
        rec = alignment.score * 100 / len(text_ref.split())
        pre = alignment.score * 100 / len(text_pred.split())
    else:
        alignment = []
        rec, pre = -1, -1

    return alignment, rec, pre


def adjust_asr_results(asr_results, second):
    adjusted_results = []
    i = 0
    for j in range(0, len(second)):
        if asr_results[i]['word'].lower() == second[j]:
            adjusted_results.append(asr_results[i]) 
            i += 1
        else:
            if adjusted_results[j-1]['word'] == '-':
                adjusted_results.append(adjusted_results[j-1])
            else:
                k = adjusted_results[j-1]['et']
                l = asr_results[i]['st']
                mean = (k+l) / 2
                adjusted_results.append({"word": second[j], "st": mean,
                                         "et": mean})
    return adjusted_results


def calculate_score_after_alignment(A, B):
    total_score = 0.0
    k = len(A)
    ref_words = 0
    asr_words = 0
    for j in range(0, k):
        if A[j] != '-':
            ref_words += 1
        if B[j] != '-':
            asr_words += 1
        if A[j] == B[j]:
            total_score += 1.0
        elif B[j] != '-' and A[j] != '-':
            total_score = total_score + Levenshtein.ratio(A[j], B[j]) 
    if asr_words==0 or ref_words==0:
        pre=0.0
        rec=0.0
    else:
        rec=total_score*100/ref_words
        pre=total_score*100/asr_words
    return rec, pre


def windows(first, second, adjusted_results, length, step):
    if step == 0:
        raise ValueError("Parameter 'm' can't be 0")
    i = length  #center of window
    k = len(second)-1
    recall_list=[]
    precision_list=[]
    while i + length < adjusted_results[k]['et']:
        list_a = []
        list_b = []
        for j in range(0, len(second)):
            bottom = i - length
            up = i + length
            if adjusted_results[j]['st'] >= bottom and \
                    adjusted_results[j]['st'] <= up:
                list_a.append(first[j])
                list_b.append(second[j])
        print(list_a)
        print(list_b)
        rec, pre = calculate_score_after_alignment(list_a, list_b)
        recall_list.append({"x": i,"y":rec})
        precision_list.append({"x":i,"y":pre})
        print('Recall score from', "%.1f" % (i-length),
              'sec', 'to', "%.1f" %(i+length), 'sec', 'is:', rec, '%')
        print('Precision score from', "%.1f" % (i-length), 'sec', 'to', "%.1f"
              % (i+length), 'sec', 'is:', pre, '%')
        if rec==0 and pre==0:
            print('Avarage score:',0.0,'%')
        else:
            print('Avarage score:',"%.1f" % (2*rec*pre/(rec+pre)),'%')
        i = i+step
    return(recall_list,precision_list)
        
