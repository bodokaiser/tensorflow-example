run(fullfile('matconvnet', 'matlab', 'vl_compilenn'));
run(fullfile('matconvnet', 'matlab', 'vl_setupnn'));

vl_simplenn_display(cnn_model())

addpath(fullfile('model'));
addpath(fullfile('model', 'init'));
addpath(fullfile('ioutil'));

dataset.images = read_minc('../mnibite/minc/13_mr.mnc');
dataset.labels = read_minc('../mnibite/minc/13_us.mnc');

addpath(fullfile('matconvnet', 'examples'));