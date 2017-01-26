function [images, labels] = batch(dataset, batch)

images = permute(dataset.images.data(:, :, batch), [1 2 4 3]);
labels = permute(dataset.images.labels(:, :, batch), [1 2 4 3]);

end

