import os
from stroke_train_function import stroke_train_img_segmentation
from stroke_test_function_with_predictions import stroke_test
from stroke_testdice import stroke_test_metrics
from create_fig_for_model import *
from sklearn.model_selection import KFold, train_test_split

# main input to function, change if needed
dir_of_train = '/Users/jk1/temp/yunet_test/hd5_test'  # data folder that contains training data
dir_of_test = '/Users/jk1/temp/yunet_test/hd5_test'  # data folder that contains testing data, since we are using 5-fold crossvalidation, its the same folder here.
list_all = np.array([f for f in os.listdir(dir_of_train) if os.path.isdir(os.path.join(dir_of_train, f))])
dir_source = '/Users/jk1/temp/yunet_test/nifti_dataset'  # source image that used to overlap the model output and calculate AUC / dice
ext_data = 'hdf5'
out_main_dir = '/Users/jk1/temp/yunet_test/'
log_dir = os.path.join(out_main_dir, 'log/')
dir_ckpt = os.path.join(out_main_dir, 'checkpoints')
predictions_dir = os.path.join(out_main_dir, 'model_predictions')
num_epochs = 1
image_dim = (79, 80, 96)
num_contrast_input = 4  # how many image contrast are fed. if input is DWI and ADC, then it's 2 contrasts. in this example, the input has 3 contrasts.
mask_contrast = 10  # default is 0, use DWI as mask; 10 is ignored
gpu = '1'  # setup which GPU you want to use
lower_lim = 0  # the limit tells the test function to output only part of the image. usually corresponding to the dimension in preprocessed h5 files. lower limit for the slices to include in an image volume
upper_lim = 79  # upper limit of the image volume
path_generic_brain_mask = '/data/brain_mask.nii'  # if no subject-specific brain mask is found, generic MNI template is used in the test to calculate AUC/dice etc.
follow_up_image_name = 'PCT'  # use PCT as mock image for now
batch_size = 16
lr_init = 0.0005
num_conv_per_pooling = 2
num_poolings = 3
num_of_aug = 1  # 1=not include mirrored image, 2=include mirrored image
model_select = 'dropout'
model_name_ori = 'test_model'  # change every time you train a new model.
output_path = os.path.join(predictions_dir, model_name_ori)
random_state = 42
n_folds = 2
if not os.path.exists(predictions_dir):
    os.mkdir(predictions_dir)
if not os.path.exists(output_path):
    os.mkdir(output_path)

loss_mode = 'bycase'

cv = 0
train_val_test_fold_indices = []  # save indices used in every split
kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
for train_index, temp_index in kf.split(list_all):
    subj_list_train = list_all[train_index]
    val_index, test_index = train_test_split(temp_index, test_size=0.5, random_state=random_state)
    train_val_test_fold_indices.append((train_index, val_index, test_index))
    val_list, subj_list_test = list_all[val_index], list_all[test_index]
    model_name = model_name_ori + '_fold{0}'.format(cv)
    filename_checkpoint = 'model_stroke_' + model_name + '.hdf5'
    filename_model = 'model_stroke_' + model_name + '.json'
    # training
    stroke_train_img_segmentation(dir_of_train, subj_list_train, lr_init=lr_init, loss_mode=loss_mode, gpu=gpu,
                                  model_select=model_select, val_list=val_list, extension_data=ext_data,
                                  log_dir=log_dir, dir_ckpt=dir_ckpt, filename_checkpoint=filename_checkpoint,
                                  filename_model=filename_model, num_epochs=num_epochs,
                                  num_contrast_input=num_contrast_input,
                                  batch_size=batch_size, num_of_aug=num_of_aug,
                                  num_conv_per_pooling=num_conv_per_pooling,
                                  num_poolings=num_poolings, shape_px_width=image_dim[1], shape_px_height=image_dim[2])
    # generate output for test cases
    stroke_test(dir_stroke=dir_of_test, dir_source=dir_source, subj_list=subj_list_test, model_name=model_name,
                dir_ckpt=dir_ckpt, filename_model=filename_model, filename_checkpoint=filename_checkpoint,
                num_contrast_input=num_contrast_input, output_path=output_path, lower_lim=lower_lim,
                upper_lim=upper_lim, followup_image_name='PCT', shape_px_width=image_dim[1],
                shape_px_height=image_dim[2])
    cv += 1
#
threshold_true = 0.5  # thresholding has been done in preprocess. if preprocess is already 0.9, try low value here.
rangelist = [0.4, 0.5, 0.6]  # you can test difference thresholds in the output and generate dice and other metrics

median_metrics = {'thres': [], 'auc': [], 'precision': [], 'recall': [], 'specificity': [], 'dice': [],
                  'volume_difference': [], 'volume_predicted': [], 'abs_volume_difference': []}
for thres in rangelist:
    print(model_name_ori)
    print('true:', threshold_true, 'prediction', thres)
    summary_list = {'subject': [], 'auc': [], 'precision': [], 'recall': [], 'specificity': [], 'dice': [],
                    'volume_difference': [], 'volume_predicted': [], 'abs_volume_difference': []}
    datawrite = False
    for cv in range(0, n_folds):
        if thres == 0.5:
            datawrite = True
        subj_list_test = list_all[train_val_test_fold_indices[cv][2]]
        model_name = model_name_ori + '_fold{0}'.format(cv)
        result_list, result_list_all, fpr, tpr, thresholds = stroke_test_metrics(
            printout=False, dir_result=output_path, dir_stroke=dir_source,
            path_generic_brain_mask=path_generic_brain_mask,
            subj_list=subj_list_test, model_name=model_name, mask_contrast=mask_contrast,
            threshold_true=threshold_true, threshold_pred=thres, lower_lim=lower_lim, upper_lim=upper_lim,
            savedata=datawrite, dwimask=False)
        # create roc figure
        if thres == 0.5:
            create_roc(fpr, tpr, result_list_all['auc'], output_path, thresholds, figname='fold{0}_roc.png'.format(cv),
                       tablename='fold{0}_roc.h5'.format(cv), datawrite=True)
        for key in summary_list:
            summary_list[key] += result_list[key]
    printout_list = ['mean', np.median(summary_list['dice']), np.median(summary_list['auc']),
                     np.median(summary_list['precision']), np.median(summary_list['recall']),
                     np.median(summary_list['specificity']), np.median(summary_list['volume_difference']),
                     np.median(summary_list['volume_predicted'])]

    for key_mean_metrics in median_metrics:
        if key_mean_metrics != 'thres':
            median_metrics[key_mean_metrics] += [
                [np.percentile(summary_list[key_mean_metrics], 25), np.median(summary_list[key_mean_metrics]),
                 np.percentile(summary_list[key_mean_metrics], 75)]]
        else:
            median_metrics[key_mean_metrics] += [[thres, thres, thres]]
    print(printout_list)
    save_dict(summary_list, output_path, filename='thres_gt' + str(thres) + '.csv', summary=True)
