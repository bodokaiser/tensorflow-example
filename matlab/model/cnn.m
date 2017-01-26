function net = cnn()

net.layers = {};
net.layers{end+1} = struct(...
    'name', 'conv', ...
    'type', 'conv', ...
    'pad', [1 1 1 1], ...
    'weights', {xavier([3 3 1 3])});
net.layers{end+1} = struct(...
    'name', 'conn', ...
    'type', 'conv', ...
    'weights', {xavier([1 1 3 1])});
net.layers{end+1} = struct(...
    'name', 'loss', ...
    'type', 'pdist', ...
    'noRoot', true, ...
    'p', 2);

net.meta.inputSize = [393 465 1];
net.meta.trainOpts.learingRate = 0.001;
net.meta.trainOpts.numEpochs = 10;
net.meta.trainOpts.batchSize = 10;

net = vl_simplenn_tidy(net);

end