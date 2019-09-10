from DataSet import DataSet
from CNN import CNN

segment_size = 128
n_filters = 196
n_channels = 6
epochs = 1000
batch_size = 600
learning_rate = 0.0001
keep_prob = 0.5
eval_iter = 10
filters_size = 16
n_classes = 6
pathDataset = 'datasets/uci_raw_data'
ds=DataSet(pathDataset,'l')
shelveDataFile=ds.PreparingData(segment_size=segment_size)
cnn=CNN(shelveDataFile,segment_size=segment_size, n_filters=n_filters,
                 n_channels=n_classes, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 keep_prob=keep_prob, eval_iter=eval_iter, filters_size=filters_size, n_classes=n_classes)
cnn.RunAndAccuracy()