import logging
from src.data_representation.Examples import load_Examples_from_file

def run_model_training_main(filename, train_config, examples=None, models=None, cluster=None, no_clusters=None):
    logging.debug("model_training.Run_model_training started main")
    if examples is None:
        try:
            examples = load_Examples_from_file(train_config["data_path"] + filename)
        except Exception as Argument:
            logging.error("Could not open file")

    logging.debug("model_training.Run_model_training ended main")
#
#
# def save_ts_model(model, examples):
#     try:
#         today = date.today().strftime("%b-%d-%Y")
#         Path("data/" + today).mkdir(parents=True, exist_ok=True)
#         Path("data/{}/{}".format(today, "model")).mkdir(parents=True, exist_ok=True)
#         model.to_pickle("data/{}/{}/{}".format(today, "model", str(model.__class__.__name__)))
#         with open("data/{}/{}/{}_{}".format(today, "model", str(model.__class__.__name__), "data"), "wb") as pkl_file:
#             pickle.dump(examples, pkl_file)
#
#     except Exception as Argument:
#         logging.error("Saving model file failed with following message:")
#         logging.error(str(Argument))