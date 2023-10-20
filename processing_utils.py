JCL_PUNCTUATION = [',','.','!','?','/','#','@','(',')','{','}']


def _remove_punctuation_jcl(text):
    for p in JCL_PUNCTUATION:
        text = text.replace(p, '')
    return text
