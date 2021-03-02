def is_list_of(seq, expected_type):
    if not isinstance(seq, list):
        return False
    if len(seq) == 0:
        return False
    if not isinstance(seq[0], expected_type):
        return False
    return True


def test_is_list_of():
    seq = [1, 2, 3]
    expected_type = int
    print(is_list_of(seq, int))

    seq = [(1), (2), (3)]
    expected_type = tuple
    print(is_list_of(seq, tuple))
    print(is_list_of(seq, float))


if __name__ == '__main__':
    test_is_list_of()
