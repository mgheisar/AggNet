import os

# path = "/nfs/nas4/marzieh/marzieh/cfp/cfpf-dataset/Data/Images/"
# for i in range(1, 500):
#     os.system('rm -r ' + path + '{:03d}'.format(i + 1) + '/frontal')
#     os.system('cp ' + path + '{:03d}'.format(i + 1) + '/profile/*.jpg ' + path + '{:03d}'.format(i + 1)+'/')
#     os.system('rm -r ' + path + '{:03d}'.format(i + 1) + '/profile')

path = '/nfs/nas4/marzieh/marzieh/AR_dataset'
for i in range(50):
    os.system('mkdir ' + path + '/{:03d}'.format(i + 1))
    os.system('mv ' + path + '/M-{:03d}-'.format(i + 1) + '*.bmp ' + path + '/{:03d}'.format(i + 1))
    os.system('rm ' + path + '/M-{:03d}-'.format(i + 1) + '*')

    os.system('mkdir ' + path + '/{:03d}'.format(i + 51))
    os.system('mv ' + path + '/W-{:03d}-'.format(i + 1) + '*.bmp ' + path + '/{:03d}'.format(i + 51))
    os.system('rm ' + path + '/W-{:03d}-'.format(i + 1) + '*')



