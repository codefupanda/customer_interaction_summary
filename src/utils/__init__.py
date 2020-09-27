# -*- coding: utf-8 -*-

def clean(nlp, strng):
    return " ".join([w.text for w in nlp(strng) if not (w.is_stop or w.is_punct)])
