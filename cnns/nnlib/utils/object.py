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

    def items(self):
        '''
        Enables enumeration via foo.items()
        '''
        return self.__dict__.items()
