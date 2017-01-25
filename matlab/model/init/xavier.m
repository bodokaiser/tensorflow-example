function init = xavier( shape )

scale = sqrt(2/prod(shape(1:3)));

init = {scale*randn(shape, 'single'), zeros(shape(4), 'single')};

end

