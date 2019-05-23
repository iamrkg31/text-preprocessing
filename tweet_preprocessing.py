import re
import html
import pandas as pd
import preprocessor as p
from nltk.tokenize import TreebankWordTokenizer

p.set_options(p.OPT.URL, p.OPT.NUMBER, p.OPT.MENTION)


def remove_unicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text


def replace_html_elements(text):
    """ Replaces html elements to standard word"""
    text = html.unescape(text)
    return text


def _tokenize(text):
    """ Tokenizes string, does not consider $ when tokenizing"""
    tokenizer = TreebankWordTokenizer()
    tokenizer.PUNCTUATION = [
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'[;@#%&]'), r' \g<0> '),
        (
            re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
            r'\1 \2\3 ',
        ),
        (re.compile(r'[?!]'), r' \g<0> '),
        (re.compile(r"([^'])' "), r"\1 ' "),
    ]
    return tokenizer.tokenize(text)


def extract_valid_words(tokens):
    """ Extracts the words relevant for analysis"""
    tokens = [tok for tok in tokens if re.search("[a-z0-9\$]+(-|'|)[a-z0-9\$]+", tok) or re.search("([a-z0-9\$]+)", tok)]
    return tokens


def clean_text(text):
    """ Cleans the text for analysis"""
    text = text.lower()
    text = re.sub("\$url\$", "$URL$", text)
    text = remove_unicode(text)
    text = replace_html_elements(text)
    text = p.tokenize(text)
    text = re.sub("^ *rt +", "", text)
    text = re.sub("\$NUMBER\$", " $NUMBER$ ", text)
    tokens = _tokenize(text)
    tokens = extract_valid_words(tokens)
    return " ".join(tokens)


df = pd.read_csv("data/test.csv")
df["cleaned_text"] = df["tweet"].apply(lambda x: clean_text(str(x)))
df.to_csv("data/cleaned_data.csv", index=False)
