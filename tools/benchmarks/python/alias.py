# Based on code found here:
# "http://code.activestate.com/recipes/
#      577659-decorators-for-adding-aliases-to-methods-in-a-clas/"


def alias(*aliases):
    def aliaser(f):
        f._aliases = set(aliases)
        return f

    return aliaser


def aliased(aliased_class):
    for name, method in list(aliased_class.__dict__.items()):
        if hasattr(method, "_aliases"):
            for alias in list(method._aliases - set(aliased_class.__dict__)):
                setattr(aliased_class, alias, method)

    return aliased_class
