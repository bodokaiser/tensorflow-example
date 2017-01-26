function init = xavier(shape)

init = {sqrt(2/prod(shape(1:3)))*randn(shape, 'single'), ...
    zeros(shape(4), 'single')};

end

