class Config:
    def __init__(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.__dict__})"

    def save(self, outdir):
        pass
