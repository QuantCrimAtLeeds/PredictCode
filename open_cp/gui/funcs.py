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


_null_logger = None

def null_logger():
    global _null_logger
    if _null_logger is None:
        _null_logger = NullLogger()
    return _null_logger


class NullLogger():
    def debug(self, *args, **kwargs):
        pass
