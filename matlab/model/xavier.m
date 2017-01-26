function init = xavier(shape)

init = {sqrt(2/prod(shape(1:3)))*randn(shape, 'double'), ...
    zeros(shape(4), 1, 'double')};

end

