function net = cnn_model()
% constructs matconvnet cnn model for mnibite dataset

net.layers = {
   struct(...
       'type', 'conv', ...
       'weights', {{randn(3,3,1,3, 'single'), zeros(1,3, 'single')}}, ...
       'stride', 1, 'pad', 0)
    struct(...
        'type', 'conv', ...
        'weights', {{randn(1,1,1,3, 'single'), zeros(1,3, 'single')}}, ...
        'stride', 1)
    struct(...
        'type', 'l2normloss');
};

net.meta.inputSize = [394 466 1];
net.meta.trainOpts.learingRate = 0.001;
net.meta.trainOpts.numEpochs = 10;
net.meta.trainOpts.batchSize = 100;

net = vl_simplenn_tidy(net);

end