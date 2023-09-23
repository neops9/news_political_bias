import  re

EMAIL_REGEX = re.compile(r"[^@]+@[^@]+\.[^@]+")
URL_REGEX = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

TIME_REGEX = re.compile(r"[0-9]{1,2}:[0-9]{1,2}(:[0-9]{1,2})?")
DATE_REGEX = re.compile(r"[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}")
SHORT_YEAR_REGEX = re.compile(r"'[0-9]{2}")
POSITION_REGEX = re.compile(r"[0-9]+th")

NUMBER_REGEX = re.compile(r"[-,/\.0-9]+")
PUNCT_REGEX = re.compile(r"[-_=\.!?]+")


def is_number(s):
    return NUMBER_REGEX.fullmatch(s) is not None


def is_mail(s):
    return EMAIL_REGEX.fullmatch(s) is not None


def is_url(s):
    return URL_REGEX.fullmatch(s) is not None


def is_time(s):
    return TIME_REGEX.fullmatch(s) is not None


def is_date(s):
    return DATE_REGEX.fullmatch(s) is not None or SHORT_YEAR_REGEX.fullmatch(s) is not None


def is_position(s):
    return s == "1st" or s == "2nd" or POSITION_REGEX.fullmatch(s) is not None


def is_punct(s):
    return PUNCT_REGEX.fullmatch(s) is not None


def is_unk(s):
    return False


NORMALIZER_DICT = {
    "*NUM*": is_number,
    "*MAIL*": is_mail,
    "*URL*": is_url,
    "*TIME*": is_time,
    "*DATE*": is_date,
    "*UNK*": is_unk,
}


class Dict:
    def __init__(self, word_to_id=None, unk_idx=None):
        self.word_to_id = word_to_id
        self.unk_idx = unk_idx

    def __len__(self):
        return len(NORMALIZER_DICT) + 1

    def to_id(self, word):
        for idx, (key, f) in enumerate(NORMALIZER_DICT.items()):
            if f(word):
                return idx + 1

            if key == "*UNK*":
                if self.word_to_id.get(word.lower(), self.unk_idx) == self.unk_idx:
                    return idx + 1

        return 0


def normalize(word):
    word = word.lower()

    if is_punct(word):
        return False, word[0]

    for key, f in NORMALIZER_DICT.items():
        if f(word):
            return True, key

    return False, word