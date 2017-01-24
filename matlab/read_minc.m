function image = read_minc( filename )
% reads volume from MINC-2.0 format and normalizes to [0,1]
range = h5readatt(filename, '/minc-2.0/image/0/image', 'valid_range');
image = h5read(filename, '/minc-2.0/image/0/image');
image = double(image - range(1)) / sum(abs(range));
end