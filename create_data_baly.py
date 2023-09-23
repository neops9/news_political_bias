from network.data import write_baly_data

write_baly_data("/data/jsons_segmented/", "/Article-Bias-Prediction/data/splits/media/train.tsv", "train", "/data/train.pkl")
write_baly_data("/data/jsons_segmented/", "/Article-Bias-Prediction/data/splits/media/valid.tsv", "dev", "/data/dev.pkl")
write_baly_data("/data/jsons_segmented/", "/Article-Bias-Prediction/data/splits/media/test.tsv", "test", "/data/test.pkl")

