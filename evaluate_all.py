import math
import subprocess

EPOCHS = 30

LOGGING_STEPS = 1
SUMMARY_STEPS = 1
CHECKPOINT_STEPS = 50

def build_args(mode, options, epoch):
    outdir = 'patch{}-filter{}'.format(options['patch_size'],
        options['filter_size'])

    args = []
    args.append('--mode={0}'.format(mode))
    args.append('--name={0}-epoch{1}'.format(mode, epoch))
    args.append('--outdir=/tmp/mrtous/{}'.format(outdir))
    args.append('--records=data/{}.tfrecord'.format(mode))
    args.append('--num_epochs=1')
    args.append('--num_threads=8')
    args.append('--batch_size={}'.format(options['batch_size']))
    args.append('--patch_size={}'.format(options['patch_size']))
    args.append('--filter_size={}'.format(options['filter_size']))
    args.append('--logging_steps={}'.format(LOGGING_STEPS))
    args.append('--summary_steps={}'.format(SUMMARY_STEPS))
    args.append('--checkpoint_steps={}'.format(CHECKPOINT_STEPS))

    return args

for patch_size in [5, 7, 9, 10, 12, 13, 15, 17, 19, 20, 22, 25, 27, 30, 33]:
    for filter_size in filter(lambda v: v < .7*patch_size, [3, 6, 9, 12, 15, 18]):
        for epoch in range(0, EPOCHS+1):
            options = {
                'batch_size': 100,
                'patch_size': patch_size,
                'filter_size': filter_size,
            }

            subprocess.check_call(['python3', 'evaluate.py',
                *build_args('train', options, epoch)])
            subprocess.check_call(['python3', 'evaluate.py',
                *build_args('test', options, epoch)])
            subprocess.check_call(['python3', 'evaluate.py',
                *build_args('validation', options, epoch)])