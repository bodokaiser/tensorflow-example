import math
import subprocess

EPOCHS = 30

LOGGING_STEPS = 1
SUMMARY_STEPS = 1
CHECKPOINT_STEPS = 50

def build_args(mode, options, epoch):
    outdir = ''
    outdir += 'batch{}-'.format(options['batch_size'])
    outdir += 'patch{}-'.format(options['patch_size'])
    outdir += 'filter{}-'.format(options['filter_size'])
    outdir += 'threshold{}'.format(options['threshold'])

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
    args.append('--threshold={}'.format(options['threshold']))
    args.append('--logging_steps={}'.format(LOGGING_STEPS))
    args.append('--summary_steps={}'.format(SUMMARY_STEPS))
    args.append('--checkpoint_steps={}'.format(CHECKPOINT_STEPS))

    return args

for patch_size in [5, 10, 15, 20, 25, 30, 35]:
    max_filter_size = math.ceil(.5*patch_size)

    for filter_size in range(3, max_filter_size, 2):
        max_threshold = math.floor(.8*patch_size)

        for threshold in range(0, max_threshold, 5):
            for epoch in range(0, EPOCHS+1):
                options = {
                    'batch_size': 100,
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