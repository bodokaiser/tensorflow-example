function [images, labels] = batch(dataset, batch)

images = dataset.images.data(:, :, 1, batch);
labels = dataset.images.labels(:, :, 1, batch);

end

