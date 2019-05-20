class Object(object):
    '''
    Creates an object for simple key/value storage; enables access via
    object or dictionary syntax (i.e. obj.foo or obj['foo']).
    '''

    def __init__(self, **kwargs):
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

    def __getitem__(self, prop):
        '''
        Enables dict-like access, ie. foo['bar']
        '''
        return self.__getattribute__(prop)

    def __str__(self):
        '''
        String-representation as newline-separated string useful in print()
        '''
        state = [f'{attr}: {val}' for (attr, val) in self.__dict__.items()]
        return '\n'.join(state)

    def get_str(self, delimiter=";"):
        state = [f'{attr}{delimiter}{val}' for (attr, val) in
                 self.__dict__.items()]
        return delimiter.join(state)

    def _get_sorted_keys(self):
        keys = self.__dict__.keys()
        sorted_keys = sorted(keys)
        return sorted_keys

    def get_str_sorted(self, delimiter=";"):
        sorted_keys = self._get_sorted_keys()
        state = [f'{attr}{delimiter}{self.__getitem__(attr)}' for attr in
                 sorted_keys]
        return delimiter.join(state)

    def get_attrs(self, delimiter=";"):
        attrs = [str(attr) for attr in self.__dict__.keys()]
        return delimiter.join(attrs)

    def get_attrs_sorted(self, delimiter=";"):
        sorted_keys = self._get_sorted_keys()
        attrs = [str(attr) for attr in sorted_keys]
        return delimiter.join(attrs)

    def get_vals(self, delimiter=";"):
        vals = [str(val) for val in self.__dict__.values()]
        return delimiter.join(vals)

    def get_vals_sorted(self, delimiter=";"):
        sorted_keys = self._get_sorted_keys()
        vals = [str(self.__getitem__(key)) for key in sorted_keys]
        return delimiter.join(vals)

    def add(self, other, prefix="", suffix=""):
        for attr, val in other.__dict__.items():
            new_attr = prefix + attr + suffix
            if new_attr in self.__dict__:
                raise Exception(f"The attribute: {new_attr} already exists.")
            self.__setattr__(new_attr, val)

    def items(self):
        '''
        Enables enumeration via foo.items()
        '''
        return self.__dict__.items()
