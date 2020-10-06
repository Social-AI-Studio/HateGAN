import json
import pickle
import re
import string

from nltk.tokenize import TweetTokenizer
import numpy as np

# Aphost lookup dict

fill = {"ain't": "is not", "aren't": "are not", "can't": "cannot",
        "can't've": "cannot have", "'cause": "because", "could've": "could have",
        "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
        "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
        "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
        "he'll've": "he he will have", "he's": "he is", "how'd": "how did",
        "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
        "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
        "I'll've": "I will have", "I'm": "I am", "I've": "I have",
        "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
        "i'll've": "i will have", "i'm": "i am", "i've": "i have",
        "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
        "it'll": "it will", "it'll've": "it will have", "it's": "it is",
        "let's": "let us", "ma'am": "madam", "mayn't": "may not",
        "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
        "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
        "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
        "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
        "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
        "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
        "she's": "she is", "should've": "should have", "shouldn't": "should not",
        "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
        "this's": "this is",
        "that'd": "that would", "that'd've": "that would have", "that's": "that is",
        "there'd": "there would", "there'd've": "there would have", "there's": "there is",
        "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
        "they'll've": "they will have", "they're": "they are", "they've": "they have",
        "to've": "to have", "wasn't": "was not", "we'd": "we would",
        "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
        "we're": "we are", "we've": "we have", "weren't": "were not",
        "what'll": "what will", "what'll've": "what will have", "what're": "what are",
        "what's": "what is", "what've": "what have", "when's": "when is",
        "when've": "when have", "where'd": "where did", "where's": "where is",
        "where've": "where have", "who'll": "who will", "who'll've": "who will have",
        "who's": "who is", "who've": "who have", "why's": "why is",
        "why've": "why have", "will've": "will have", "won't": "will not",
        "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
        "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
        "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
        "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
        "you'll've": "you will have", "you're": "you are", "you've": "you have", "u.s.": "united states",
        "#lol": "laughing out loud", "#lamo": "laughing my ass off", "#rof": "rolling on the floor laughing",
        "#covfefe": "ironic", "wtf": "what the fuck", "#wtf": "what the fuck",
        "tbh": "to be honest"}

slang = {
    "4ward": "forward",
    "brb": "be right back",
    "b4": "before",
    "bfn": "bye for now",
    "bgd": "background",
    "btw": "by the way",
    "br": "best regards",
    "clk": "click",
    "da": "the",
    "deet": "detail",
    "deets": "details",
    "dm": "direct message",
    "f2f": "face to face",
    "ftl": " for the loss",
    "ftw": "for the win",
    "f**k": "fuck",
    "f**ked": "fucked",
    "b***ch": "bitch",
    "kk": "cool cool",
    "kewl": "cool",
    "smh": "so much hate",
    "yaass": "yes",
    "a$$": "ass",
    "bby": "baby",
    "bc": "because",
    "coz": "because",
    "cuz": "because",
    "cause": "because",
    "cmon": "come on",
    "cmonn": "come on",
    "dafuq": "what the fuck",
    "dafuk": "what the fuck",
    "dis": "this",
    "diss": "this",
    "ma": "my",
    "dono": "do not know",
    "donno": "do not know",
    "dunno": "do not know",
    "fb": "facebook",
    "couldnt": "could not",
    "n": "and",
    "gtg": "got to go",
    "yep": "yes",
    "yw": "you are welcome",
    "im": "i am",
    "youre": "you are",
    "hes": "he is",
    "shes": "she is",
    "theyre": "they are",
    "af": "as fuck",
    "fam": "family",
    "fwd": "forward",
    "ffs": "for fuck sake",
    "fml": "fuck my life",
    "lol": "laugh out loud",
    "lel": "laugh out loud",
    "lool": "laugh out loud",
    "lmao": "laugh my ass off",
    "lmaoo": "laugh my ass off",
    "omg": "oh my god",
    "oomg": "oh my god",
    "omgg": "oh my god",
    "omfg": "oh my fucking god",
    "stfu": "shut the fuck up",
    "awsome": "awesome",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "ily": "i love you",
    "ilyy": "i love you",
    "ikr": "i know right",
    "ikrr": "i know right",
    "idk": "i do not know",
    "jk": "joking",
    "lmk": "let me know",
    "nsfw": "not safe for work",
    "hehe": "haha",
    "tmrw": "tomorrow",
    "yt": "youtube",
    "hahaha": "haha",
    "hihi": "haha",
    "pls": "please",
    "ppl": "people",
    "wtf": "what the fuck",
    "wth": "what teh hell",
    "obv": "obviously",
    "nomore": "no more",
    "u": "you",
    "ur": "your",
    "wanna": "want to",
    "luv": "love",
    "imma": "i am",
    "&": "and",
    "thanx": "thanks",
    "til": "until",
    "till": "until",
    "thx": "thanks",
    "pic": "picture",
    "pics": "pictures",
    "gp": "doctor",
    "xmas": "christmas",
    "rlly": "really",
    "boi": "boy",
    "boii": "boy",
    "rly": "really",
    "whch": "which",
    "awee": "awsome",  # or maybe awesome is better
    "sux": "sucks",
    "nd": "and",
    "fav": "favourite",
    "frnds": "friends",
    "info": "information",
    "loml": "love of my life",
    "bffl": "best friend for life",
    "gg": "goog game",
    "xx": "love",
    "xoxo": "love",
    "thats": "that is",
    "homie": "best friend",
    "homies": "best friends"
}


def normalize_word(word):
    temp = word
    while True:
        w = re.sub(r"([a-zA-Z])\1\1", r"\1\1", temp)
        if (w == temp):
            break
        else:
            temp = w
    return w


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def refer_normalize(tokens):
    words = []
    for idx in range(len(tokens)):
        if idx + 1 != len(tokens) and tokens[idx].startswith("@") and tokens[idx + 1].startswith("@"):
            continue
        else:
            words.append(tokens[idx])
    return words


def clean_text(text):
    # remove url
    text = re.sub(r'http\S+', '', text)

    # fixing apostrope
    text = text.replace("’", "'")

    # remove &amp;
    text = text.replace('&amp;', 'and ')

    # remove \n
    text = re.sub("\\n", "", text)

    # remove leaky elements like ip,user
    text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", text)

    # (')aphostophe  replacement (ie)   you're --> you are
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    # tokenizer = TweetTokenizer()
    tokens = text.split()
    tokens = refer_normalize(tokens)
    tokens = [fill[word] if word in fill else word for word in tokens]
    tokens = [fill[word.lower()] if word.lower() in fill else word for word in tokens]
    tokens = [slang[word] if word in slang else word for word in tokens]
    tokens = [slang[word.lower()] if word.lower() in slang else word for word in tokens]
    # tokens = [normalize_word(word) for word in tokens]
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in tokens if ch not in exclude)
    text = ' '.join(tokens)

    # removing usernames
    text = ' '.join(re.sub("(@[A-Za-z0-9_]+)", "", text).split())
    text = re.sub("'s", "", text)
    text = re.sub("'", "", text)

    # emoji remover
    text = remove_emoji(text)

    text = re.sub("&#\S+", "", text)
    text = re.sub("RT :", "", text)

    # remove non-ascii
    # text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # text = re.sub(r'[^\w\s]', '', text)

    # to lower
    # text = text.lower()
    return text


def dump(data):
    def default(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        raise TypeError('Unknown type:', type(obj))

    json.dumps(data, default=default)


if __name__ == '__main__':
    txt = "RT @QuickTake: The meeting was at full capacity throughout the 2 hours and 30 minutes. This couple in India took their traditional weddi…"
    re = clean_text(txt)
    print(re)
