function [images, labels] = batch(dataset, batch)

images = single(dataset.images.data(:, :, 1, batch));
labels = single(dataset.images.labels(:, :, 1, batch));

end

