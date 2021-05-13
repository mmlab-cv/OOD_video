from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51


def get_training_set(opt, video_path, annotation_path, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['ucf101', 'hmdb51']
    if opt.dataset == 'ucf101':
        video_path = os.path.join(args.root_path, video_path)
        training_data = UCF101(
            video_path,
            annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'hmdb51':
        training_data = HMDB51(
            video_path,
            annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_treining_set(opt, annotation_path, video_path, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['ucf101', 'hmdb51']
    if opt.dataset == 'ucf101':
        validation_data = UCF101(
            video_path,
            annotation_path,
            'training',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        validation_data = HMDB51(
            video_path,
            annotation_path,
            'training',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return validation_data

def get_validation_set(opt, annotation_path, video_path, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['ucf101', 'hmdb51']

    if opt.dataset == 'ucf101':
        validation_data = UCF101(
            video_path,
            annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        validation_data = HMDB51(
            video_path,
            annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return validation_data


def get_test_set(opt, annotation_path, video_path, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['ucf101', 'hmdb51']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'ucf101':
        test_data = UCF101(
            video_path,
            annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(
            video_path,
            annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)

    return test_data
