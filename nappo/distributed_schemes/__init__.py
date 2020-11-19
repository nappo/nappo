
def get_scheme_workers(scheme_name):
    """Return scheme workers from distributed scheme name"""

    if scheme_name == "3cs":
        from .scheme_3cs import Worker
        return Worker
    elif scheme_name == "3ds":
        from .scheme_3ds import Workers
        return Workers
    elif scheme_name == "2dacs":
        from .scheme_2dacs import Workers
        return Workers
    elif scheme_name == "2daca":
        from .scheme_2daca import Workers
        return Workers
    elif scheme_name == "da2cs":
        from .scheme_da2cs import Workers
        return Workers
    elif scheme_name == "dadacs":
        from .scheme_dadacs import Workers
        return Workers
    elif scheme_name == "dadaca":
        from .scheme_dadaca import Workers
        return Workers
    else:
        raise NotImplementedError