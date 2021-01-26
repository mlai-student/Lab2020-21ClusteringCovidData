#return the config as a dict to place it into the overview csv tables
def get_config_dict(config):
    dictionary = {}
    for section in config.sections():
        dictionary[section] = {}
        for option in config.options(section):
            dictionary[str(section) + " " + str(option)] = config.get(section, option)
    return dictionary
