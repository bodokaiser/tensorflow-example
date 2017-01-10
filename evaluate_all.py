import math
import subprocess

EPOCHS = 30

def build_args(mode, options, epoch):
    outdir = 'patch{}-filter{}-threshold{}'.format(options['patch_size'],
        options['filter_size'], options['threshold'])

    args = []
    args.append('--mode={0}'.format(mode))
    args.append('--name={0}-epoch{1}'.format(mode, epoch))
    args.append('--outdir=/tmp/mrtous/{}'.format(outdir))
    args.append('--records=data/{}.tfrecord'.format(mode))
    args.append('--num_epochs=1')
    args.append('--patch_size={}'.format(options['patch_size']))
    args.append('--filter_size={}'.format(options['filter_size']))

    return args

for patch_size in [5, 10, 15, 25, 35]:
    for filter_size in filter(lambda v: v < .7*patch_size, [3, 6, 9, 12]):
        for threshold in filter(lambda v: v < .8*patch_size**2, [0, 1, 2, 5, 10, 15, 20, 25]):
            for epoch in range(0, EPOCHS+1):
                options = {
                    'patch_size': patch_size,
                    'filter_size': filter_size,
                    'threshold': threshold,
                }

                subprocess.check_call(['python3', 'evaluate.py',
                    *build_args('train', options, epoch)])
                subprocess.check_call(['python3', 'evaluate.py',
                    *build_args('test', options, epoch)])
                subprocess.check_call(['python3', 'evaluate.py',
                    *build_args('validation', options, epoch)])