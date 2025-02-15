

def encode(input_string):

    if not input_string:
        return []

    count = 1
    prev = ""
    lst = []
    for character in input_string:
        if character != prev:
            if prev:
                entry = (prev, count)
                lst.append(entry)
            count = 1
            prev = character
        else:
            count += 1
    entry = (character, count)
    lst.append(entry)
    return lst


def decode(lst):
    q = ""
    for character, count in lst:
        q += character * count
    return q

# encodedResult = encode('fayezz')

# print('encodedResult is: ', encodedResult)

# decodedResult = decode(encodedResult)

# print('decodedResult is: ', decodedResult)

from hypothesis import example, given, strategies as st

from hypothesis.strategies import text


# @given(text())
# def test_decode_inverts_encode(s):
#     assert decode(encode(s)) == s



# @given(st.text())
# @example("")
# def test_decode_inverts_encode(s):
#     assert decode(encode(s)) == s


# @given(s=st.text())
# @example(s="")
# def test_decode_inverts_encode(s):
#     assert decode(encode(s)) == s

# if __name__ == "__main__":
#     test_decode_inverts_encode()

    