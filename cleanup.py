import os, shutil

print('Running Cleanup')
count = 0
for i in range(300):
    frames_dir = 'exp_frames' + str(i)
    if os.path.exists(frames_dir):
        count = count + 1
        # raise ValueError('Frames directory already exists.')
        shutil.rmtree(frames_dir)

print('Removed ' + str(count) + ' files')
