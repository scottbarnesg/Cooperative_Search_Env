import os, shutil

print('Running Cleanup')
count = 0
for i in range(301):
    frames_dir = 'exp_frames' + str(i)
    if os.path.exists(frames_dir):
        count = count + 1
        # raise ValueError('Frames directory already exists.')
        shutil.rmtree(frames_dir)

print('Removed ' + str(count) + ' folders')
count = 0
for file in os.listdir(os.getcwd()):
    if file.endswith('.png'):
        os.remove(file)
        count = count+1

print('Removed ' + str(count) + ' .png files')
