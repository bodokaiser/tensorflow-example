function Y = vl_l2loss(X, c, dzdy)

if nargin <= 2
    Y = sum((squeeze(X)'-c).^2)/2;
else
    Y = +((squeeze(X)'-c))*dzdy;
    Y = reshape(Y, size(X));
end

end

