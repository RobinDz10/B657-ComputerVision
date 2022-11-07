import regex as re
import numpy as np
#from PyDictionary import PyDictionary
import enchant
#from autocorrect import Speller
from wordfreq import word_frequency
import string

# original example function: not used
def detect_number(text_full, index):
    print(text_full.replace('\n','')+'|||||'+input+'||||||'+str(index))
    return re.match(r"\d+", input) or re.match(r"\+\d+", input)

def is_num_word(word: str) -> bool:
    n = len(word)
    m = len(re.sub("[^0-9]", "", word))
    if m > 0 and n - m <= 1:
        return True
    else:
        return False

def detect_all_num(input: list) -> bool:
    #n = len(input)
    #m = 0
    for word in input:
        if is_num_word(word) == False:
            return False

    return True

    #if m > 1 and n - m <= 1:
        #return True
    #else:
        #return False

def detect_prefix(input: list, index: int) -> bool:
    # do not consider non-num words for prefix
    if is_num_word(input[index]) == False:
        return False
    n = len(input)

    # prefix words which are sensitive/private
    phone_words = ['phone', 'contact', 'number', 'should', 'this']
    id_words = ['id', 'ssn', 'security', 'account', 'username']
    finance_words = ['salary', 'stipend', 'payment', 'balance', 'price', 'bank']
    code_words = ['code', 'key', 'password', 'passphrase']
    address_words = ['address', 'home', 'house', 'apartment']
    
    # positions in string list which are close to prefix/sensitive words
    prefix_ranges = np.zeros(n, dtype=bool)
    for prefix in phone_words + id_words + finance_words + code_words + address_words:
        for i in range(n):
            if prefix in input[i]:
                j = i - 1
                while j >= max(i-3, 0):
                    prefix_ranges[j] = True
                    j -= 1
                j = i + 1
                while j <= min(i+3, n-1):
                    prefix_ranges[j] = True
                    j += 1

    # positions of surrounding words which are numbers and consecutive to pattern
    pattern_ranges = np.zeros(n, dtype=bool)
    # search left until a non-number word
    i = index
    while i >= 0:
        if is_num_word(input[i]):
            pattern_ranges[i] = True
            i -= 1
        else:
            break
    # search right until a non-number word
    i = index
    while i < n:
        if is_num_word(input[i]):
            pattern_ranges[i] = True
            i += 1
        else:
            break

    # return True if there is overlap between prefix ranges and pattern ranges:
    return np.any(prefix_ranges & pattern_ranges)

def detect_address(input: list, index: int) -> bool:
    n = len(input)
    address_ranges = np.zeros(n, dtype=bool)
    # find the address range in input list
    for i in range(n):
        if is_num_word(input[i]):
            j = i+1
            while j < n:
                freq = word_frequency(input[j], 'en')
                if freq > 2e-4:
                    break
                j += 1
            if j - i > 3:
                address_ranges[i:j] = True
    # if pattern falls in the address range
    if address_ranges[index] == True:
        return True

    return False

def detect_code(input: list, index: int) -> bool:
    # check spelling: not working well
    #spell = Speller()
    #pattern_correct = spell(pattern_correct)

    dictionary = enchant.Dict("en_US")
    meaning = dictionary.check(input[index])
    freq = word_frequency(input[index], 'en')
    if meaning == False and freq < 1e-7:
        return True

    return False

def detect_nickname(input: list, index: int) -> bool:
    n = len(input)
    if n == 1:
        freq = word_frequency(input[index], 'en')
        if freq < 4e-5 and is_num_word(input[index]) == False:
            return True
    elif n == 2:
        freq1 = word_frequency(input[index], 'en')
        freq2 = word_frequency(input[1-index], 'en')
        if freq1 < 4e-5 and freq2 > 4e-5 and is_num_word(input[index]) == False:
            return True
    elif n == 3:
        if index == 0:
            freq1 = word_frequency(input[index], 'en')
            freq2 = word_frequency(input[index+1], 'en')
            if freq1 < 4e-5 and freq2 > 4e-5 and is_num_word(input[index]) == False:
                return True
        elif index == 1:
            freq1 = word_frequency(input[index-1], 'en')
            freq2 = word_frequency(input[index], 'en')
            freq3 = word_frequency(input[index+1], 'en')
            if freq1 > 4e-5 and freq2 < 4e-5 and freq2 > 4e-5 and is_num_word(input[index]) == False:
                return True
        elif index == 2:
            freq1 = word_frequency(input[index-1], 'en')
            freq2 = word_frequency(input[index], 'en')
            if freq1 > 4e-5 and freq2 < 4e-5 and is_num_word(input[index]) == False:
                return True

    return False

def detect_privacy(input: list, index: int) -> bool:
    # always return false if pattern is empty
    if not input[index]:
        return False

    # preprocessing: remove punctuation
    # reference: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    for i, word in enumerate(input):
        input[i] = word.translate(str.maketrans('', '', string.punctuation))
    if not input[index]:
        return False

    # preprocessing: remove empty words and update index
    # reference: https://www.geeksforgeeks.org/python-remove-empty-strings-from-list-of-strings/
    input_left = list(filter(None, input[:index]))
    index = len(input_left)
    input = list(filter(None, input))

    # preprocessing: convert to lowercase
    for i, word in enumerate(input):
        input[i] = word.lower()

    # detect if input string is of numbers only
    num = detect_all_num(input)
    if num == True:
        return True
    # detect numbers: phone, id, salary
    prefix = detect_prefix(input, index)
    if prefix == True:
        return True
    # detect address
    address = detect_address(input, index)
    if address == True:
        return True
    # detect rare words: code, key, password
    code = detect_code(input, index)
    if code == True:
        return True
    # detect nick name
    name = detect_nickname(input, index)
    if name == True:
        return True

    return False
    #result = np.asarray([num, prefix, address, code])
    #return np.any(result)
