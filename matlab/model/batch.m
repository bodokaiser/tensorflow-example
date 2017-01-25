function [images, labels] = batch( dataset, batch )

images = dataset.images(:, :, batch);
labels = dataset.labels(:, :, batch);

end

