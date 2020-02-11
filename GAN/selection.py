import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import read_img
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, '/home/alireza/Desktop/Deep-Learning/UCF101-Active-Learning')
import acquisition, kmedoids
import matlab
import matlab.engine
import random

views = ['200', '190', '041', '050', '051', '140', '130', '080', '090']

multi_pie_path = 'data/multi_PIE_crop_128/'

subjects = [x[0].split('/')[2] for x in os.walk(multi_pie_path)]

feat_dimension = 200
selected_samples = 9
selection = 'FFS'
print 'selecting ', selected_samples, ' samples in ', feat_dimension, 'dimensional space using ',  selection

list_path = 'data/multi_PIE_crop_128/selected_' + selection + '_' + str(feat_dimension) + '_' + str(selected_samples) + '.txt'
list_file = open(list_path, 'w')

for subject in subjects:
    if len(subject) == 0:
        continue

    # if subject == '201':
    #     break

    print 'Subject', subject, '...',

    subject_path = multi_pie_path + subject
    images = []
    file_paths = []
    files_views = []
    for file in os.listdir(subject_path):
        if file.endswith(".png"):
            view = file.split('_')[3]
            if view not in views:
                continue
            file_path = subject_path + '/' + file
            img = read_img(file_path).convert('L')
            images.append(np.array(img).ravel())
            file_paths.append(file_path)
            files_views.append(views.index(view))

    if len(images) < selected_samples:
        print 'Skipped!'
        continue

    if selection == 'none':
        for idx in range(len(file_paths)):
            list_file.write(file_paths[idx] + ' ' + str(files_views[idx]) + '\n')
        print 'Done!'
        continue
    elif selection == 'random':
        for idx in random.sample(range(len(file_paths)), selected_samples):
            list_file.write(file_paths[idx] + ' ' + str(files_views[idx]) + '\n')
        print 'Done!'
        continue



    images = np.array(images)

    pca = PCA(n_components=feat_dimension)
    pca.fit(images)
    images_reduced = pca.transform(images)

    if selection == 'IPM':
        selected_idx = acquisition.optimal_acquisition([], images_reduced, np.zeros(len(images_reduced)), selected_samples,
                                                       1, type='IPM')
    elif selection == 'SP':
        selected_idx = acquisition.optimal_acquisition([], images_reduced, np.zeros(len(images_reduced)), selected_samples,
                                                       1, type='SP')
    elif selection == 'MP':
        selected_idx = acquisition.optimal_acquisition([], images_reduced, np.zeros(len(images_reduced)), selected_samples,
                                                       1, type='MP')
    elif selection == 'kmedoids':
        selected_idx, _ = kmedoids.k_medoids_selection(images_reduced, selected_samples)
    elif selection == 'DS3':
        eng = matlab.engine.start_matlab()
        selected_idx = acquisition.ds3_selection(eng, images_reduced, selected_samples)
    elif selection == 'FFS':
        eng = matlab.engine.start_matlab()
        selected_idx = acquisition_mehr_ffs.ffs(eng, images_reduced, selected_samples)	
    for idx in selected_idx:
        list_file.write(file_paths[idx] + ' ' + str(files_views[idx]) + '\n')

    # plt.figure()
    # i = 1
    # for idx in selected_idx:
    #     list_file.write(file_paths[idx] + '\n')
    #     ax = plt.subplot(1, selected_samples, i)
    #     i += 1
    #     ax.imshow(read_img(file_paths[idx]))


    print 'Done!'
    # break
    # print img

plt.show()
list_file.close()
