class SEATag(object):
    def __init__(self, description, tag_number, samples, size,
            typ, parameter1, parameter2, parameter3):
        self.description = description
        self.tag_number = tag_number
        self.samples = samples
        self.size = size
        self.typ = typ
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.parameter3 = parameter3

        for attr_name in vars(self):
            if attr_name != 'description':
                attr_val = getattr(self, attr_name)
                try:
                    setattr(self, attr_name, int(attr_val, 0))
                except ValueError as ve:
                    # if it didn't convert, it wasn't an int, so just leave it
                    pass
                except TypeError as te:
                    if te.args[0] == "int() can't convert non-string with explicit base":
                        # don't bother convertingthe non-string
                        pass
                    else:
                        raise

    def __repr__(self):
        parts = ['<sea.Tag ']
        for key, value in vars(self).items():
            parts.append('{:s}={}'.format(key,
                value if value is not None else 'undefined'))
        parts.append('>')
        return ' '.join(parts)

    # At the time of writing, and forseeable future, tag numbers can be
    # used as a primary identifier for tags within any given file from an
    # SEA M*00 system. (ie If two tags share the same tag number, all of
    # their remaining properties will be identical, and if two tags have
    # different tag numbers, some, but not necessarily all, of their
    # remaining properties will have different values). With this in mind,
    # __cmp__ and __hash__ have been implemented in the simplest way
    # possible to allow SEATag objects to be used as dictionary keys.
    def __hash__(self):
        return self.tag_number

    def __cmp__(self, other):
        return self.tag_number - other.tag_number


class M200Tag(SEATag):
    def __init__(self, description, buffer_number, tag_number, samples, size,
            typ, parameter1, parameter2, parameter3, address):
        super(M200Tag, self).__init__(description, tag_number, samples, size,
                typ, parameter1, parameter2, parameter3)

        self.buffer_number = buffer_number
        self.address = address

        for attr_name in ['buffer_number', 'address']:
            attr_val = getattr(self, attr_name)
            try:
                setattr(self, attr_name, int(attr_val, 0))
            except ValueError as ve:
                # if it didn't convert, it wasn't an int, so just leave it
                pass
            except TypeError as te:
                if te.message == \
                        "int() can't convert non-string with explicit base":
                    # don't bother converting the non-string
                    pass
                else:
                    raise

    def __repr__(self):
        parts = ['<m200.Tag ']
        for key, value in vars(self).items():
            parts.append('{:s}={}'.format(key,
                value if value is not None else 'undefined'))
        parts.append('>')
        return ' '.join(parts)


class M300Tag(SEATag):
    def __init__(self, description, tag_number, samples, state, size,
            typ, parameter1, parameter2, parameter3, board, sample_offset=0):
        super(M300Tag, self).__init__(description, tag_number, samples, size,
                typ, parameter1, parameter2, parameter3)

        self.state = state
        self.board = board
        self.sample_offset = sample_offset

        for attr_name in ['state', 'sample_offset']:
            attr_val = getattr(self, attr_name)
            try:
                setattr(self, attr_name, int(attr_val, 0))
            except ValueError as ve:
                # if it didn't convert, it wasn't an int, so just leave it
                pass
            except TypeError as te:
                if te.args[0] == "int() can't convert non-string with explicit base":
                    # don't bother convertingthe non-string
                    pass
                else:
                    raise

    def __repr__(self):
        parts = ['<m300.Tag ']
        for key, value in vars(self).items():
            parts.append('{:s}={}'.format(key,
                value if value is not None else 'undefined'))
        parts.append('>')
        return ' '.join(parts)
