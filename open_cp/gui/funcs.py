"""
funcs
~~~~~

Misc functions
"""

def string_ellipse(string, maxlen):
    """Clamp the string to be no longer than the maximum length.  If the string
    is too long, we write it as "... []" where "[]" is the final part of the
    string.
    
    :param string: The input string.
    :param maxlen: The maximum length of the string.
    """
    if len(string) <= maxlen:
        return string
    
    if maxlen <= 4:
        raise ValueError("maxlen too small")
    
    return "... " + string[4-maxlen:]