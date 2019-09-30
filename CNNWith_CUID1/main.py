from DataSet import DataSet
from CNN import CNN
from AlternativeCNN import CNN
import CNNWithoutFeatures

segment_size = 128
n_filters = 196
n_channels = 6
epochs = 1000
batch_size = 200
learning_rate = 5e-4
dropout_rate = 0.05
eval_iter = 1000
filters_size = 16   
n_classes = 6
pathDataset = 'datasets/uci_raw_data'
ds = DataSet(pathDataset, 'l')
shelveDataFile = ds.PreparingData(segment_size=segment_size)
cnn = CNNWithoutFeatures.CNN(shelveDataFile, segment_size=segment_size, n_filters=n_filters,
          n_channels=n_classes, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
          dropout_rate=dropout_rate, eval_iter=eval_iter, filters_size=filters_size, n_classes=n_classes)
cnn.RunAndAccuracy()
