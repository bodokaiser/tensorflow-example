%run(fullfile('matconvnet', 'matlab', 'vl_compilenn'));
run(fullfile('matconvnet', 'matlab', 'vl_setupnn'));

addpath(fullfile('model'));
addpath(fullfile('ioutil'));
addpath(fullfile('matconvnet', 'examples'));

mr = read_minc('../mnibite/minc/13_mr.mnc');
us = read_minc('../mnibite/minc/13_us.mnc');

num_samples = size(mr, 3);
set_indices = randperm(num_samples);

mr_resized = zeros([393 465 1 378], 'single');
us_resized = zeros([393 465 1 378], 'single');

for i = 1:num_samples
    mr_resized(:, :, 1, i) = imresize(mr(i), [393 465]);
    us_resized(:, :, 1, i) = imresize(us(i), [393 465]);
end

dataset.images.data = mr_resized;
dataset.images.labels = us_resized;

dataset.images.id = 1:num_samples;
dataset.images.set = ones(1, num_samples);
dataset.images.set(set_indices(floor(.7*num_samples):1:floor(.9*num_samples))) = 2;
dataset.images.set(set_indices(floor(.9*num_samples)+1:1:end)) = 3;

trainOpts.learningRate = .001;
trainOpts.numEpochs = 10;
trainOpts.batchSize = 10;
trainOpts.errorFunction = 'none';

[net, info] = cnn_train(cnn(), dataset, @batch, trainOpts)