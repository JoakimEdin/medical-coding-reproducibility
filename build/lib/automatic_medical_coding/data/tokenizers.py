def word_tokenizer(string):
    """
    Splits a string by whitespace characters.
    Args:
        string (string): The string to be split by whitespace characters.
    Returns:
        list: The words of the string.
    """
    return string.split()


def char_tokenizer(string):
    """
    Splits a string into individual character symbols.
    Args:
        string (string): The string to be split into characters.
    Returns:
        list: The characters of the string.
    """
    return list(string)
