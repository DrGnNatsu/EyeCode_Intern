import numpy as np


def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices
def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result



def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }

    return stats


def normalize_data(data, stats):
    # Normalize to [0,1]
    diff = stats['max'] - stats['min']
    diff[diff == 0] = 1  # Avoid division by zero by setting diff to 1 where max == min
    ndata = (data - stats['min']) / diff
    
    # Normalize to [-1,1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    diff = stats['max'] - stats['min']
    diff[diff == 0] = 1  # Avoid division by zero by setting diff to 1 where max == min
    data = ndata * diff + stats['min']
    return data


# Standard ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_image_imagenet(image_array: np.ndarray) -> np.ndarray:
    """
    image_array: (N, H, W, 3) or (H, W, 3)  (i.e., channel-last) *before* you move axis,
                 or (N, 3, H, W) or (3, H, W) (channel-first) *after* you move axis.
    We just need to handle the shape carefully.
    
    Steps:
      1) scale from [0,255] to [0,1]
      2) subtract mean, divide by std (per channel)
    """

    # 1) If images are originally in [0,255], scale to [0,1]
    #    But only if you haven’t already scaled them!
    #    For example, if your original images are already float in [0,1], skip this step.
    #    Here we assume they are in [0,255].
    #    If you’ve already done `image_array = image_array / 255.0` prior to calling
    #    this function, remove this line.
    #
    # image_array /= 255.0

    # 2) Subtract mean, divide by std. This depends on shape.
    #    We'll handle (N, 3, H, W), (3, H, W), (N, H, W, 3), (H, W, 3).
    
    if image_array.ndim == 4:
        # Possibly (N, H, W, 3) or (N, 3, H, W)
        if image_array.shape[1] == 3:
            # shape = (N, 3, H, W) → channel-first
            # We can broadcast mean/std across [N, 3, H, W]
            image_array[:, 0, :, :] = (image_array[:, 0, :, :] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            image_array[:, 1, :, :] = (image_array[:, 1, :, :] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            image_array[:, 2, :, :] = (image_array[:, 2, :, :] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        else:
            # shape = (N, H, W, 3) → channel-last
            image_array[:, :, :, 0] = (image_array[:, :, :, 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            image_array[:, :, :, 1] = (image_array[:, :, :, 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            image_array[:, :, :, 2] = (image_array[:, :, :, 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    elif image_array.ndim == 3:
        # Possibly (3, H, W) or (H, W, 3)
        if image_array.shape[0] == 3:
            # shape = (3, H, W)
            image_array[0, :, :] = (image_array[0, :, :] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            image_array[1, :, :] = (image_array[1, :, :] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            image_array[2, :, :] = (image_array[2, :, :] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        else:
            # shape = (H, W, 3)
            image_array[:, :, 0] = (image_array[:, :, 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            image_array[:, :, 1] = (image_array[:, :, 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            image_array[:, :, 2] = (image_array[:, :, 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    else:
        raise ValueError(f"Expected 3D or 4D shape for image, got shape {image_array.shape}")

    return image_array

