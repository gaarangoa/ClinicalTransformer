class Config:
    def __init__(self, data):
        # If the data is a dictionary, iterate over the keys and values
        if isinstance(data, dict):
            for key, value in data.items():
                # Recursively create attributes for nested dictionaries
                setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def __getitem__(self, item):
        # This allows us to use the indexing method like `config['model']`
        return getattr(self, item)

    def __setitem__(self, key, value):
        # Allows us to set attributes via indexing like `config['model'] = value`
        setattr(self, key, value)

    def __repr__(self):
        # A simple representation for debugging purposes
        return f"Config({self.__dict__})"

    def save(self, outdir): 
        pass