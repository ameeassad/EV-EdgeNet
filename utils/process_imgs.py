
import numpy as np
import os
import sys
import glob
import cv2

class FixedDurationEventReader:
    def __init__(self, path_to_dataset, duration_ms=50.0, start_index=0):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))

        self.p = np.load(path_to_dataset + '_events_p.npy')
        self.t = np.load(path_to_dataset + '_events_t.npy')
        self.xy = np.load(path_to_dataset + '_events_xy.npy')

        self.last_stamp = None
        self.duration_s = duration_ms / 1000.0
        self.index = start_index

    def __iter__(self):
        return self

    def __next__(self):
        event_list = []
        while self.index < len(self.t):
            t, xy, p = self.t[self.index], self.xy[self.index], self.p[self.index]
            x, y = xy
            event_list.append([t, x, y, p])
            if self.last_stamp is None:
                self.last_stamp = t
            if t > self.last_stamp + self.duration_s:
                self.last_stamp = t
                event_window = np.array(event_list)
                return event_window

            self.index += 1

        raise StopIteration

def process_directory(input_dir, output_dir, width=640, height=480, duration_ms=50):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    input_dir = os.path.join(input_dir, 'samsung_mono/imo')
    # Process all directories recursively
    for root, dirs, files in os.walk(input_dir):
        if 'dataset_events_t.npy' in files:
            # Found a directory with event data, process it
            dataset = os.path.join(root, 'dataset')
            output_path = os.path.join(output_dir, os.path.relpath(root, input_dir))

            print("processing", dataset)
            print("output path: ", output_path)

            process_dataset(dataset, output_path, width, height, duration_ms)

def process_dataset(dataset, output_path, width=640, height=480, duration_ms=50.0):
    # If you want Event Masks:
    mask_data = np.load(dataset + '_mask.npz')
    dataset_info = np.load(dataset + '_info.npz', allow_pickle=True)['meta'].item()
    # If you want RGB Masks:
    # mask_data = np.load(dataset.replace("samsung_mono", "flea3_7_npz") + '_mask.npz')
    # dataset_info = np.load(dataset.replace("samsung_mono", "flea3_7_npz") + '_info.npz', allow_pickle=True)['meta'].item()
    # Load classical image data
    classical_data = np.load(dataset.replace("samsung_mono", "flea3_7_npz") + '_classical.npz')
    dataset_info_img = np.load(dataset.replace("samsung_mono", "flea3_7_npz") + '_info.npz', allow_pickle=True)['meta'].item()

    # Extract the frame info
    frame_infos = dataset_info['frames']
    frame_infos_img = dataset_info_img['frames']

    # Create a dictionary to store the masks and classical images with their timestamps as keys
    mask_dict = {}
    for frame_info in frame_infos:
        if 'gt_frame' in frame_info:
            mask_key = f"mask_{frame_info['id']:010d}"
            if mask_key in mask_data:
                mask_dict[frame_info['ts']] = mask_data[mask_key]
    classical_dict = {}
    for frame_info in frame_infos_img:
        if 'classical_frame' in frame_info:
            classical_key = f"classical_{frame_info['id']:010d}"
            if classical_key in classical_data:
                classical_dict[frame_info['ts']] = classical_data[classical_key]

    # Split the path to get the scene name
    _, scene_name = os.path.split(os.path.dirname(dataset))

    # Create events directory if not exists
    events_dir = os.path.join(os.path.dirname(output_path), 'events')
    os.makedirs(events_dir, exist_ok=True)

    # Create labels directory if not exists
    labels_dir = os.path.join(os.path.dirname(output_path), 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    # Create images directory if not exists
    images_dir = os.path.join(os.path.dirname(output_path), 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Initialize data structures
    event_counts = np.zeros((height, width, 2))
    mean_timestamps = np.zeros((height, width, 2))
    sum_of_squares = np.zeros((height, width, 2))

    # Load and process event data
    reader = FixedDurationEventReader(dataset, duration_ms=duration_ms)

    window_index = 0  # to create a unique filename for each window

    for event_window in reader:
        # Reset counters for each time window
        event_counts.fill(0)
        mean_timestamps.fill(0)
        sum_of_squares.fill(0)

        # timestamp for current window
        # get timestamp for LAST event in the window
        window_timestamp = event_window[-1][0]

        for t, x, y, p in event_window:
            t, x, y, p = float(t), int(x), int(y), int(p)
            # Update event counts
            event_counts[y, x, p] += 1

            # Update mean timestamps and sum of squares
            old_mean = mean_timestamps[y, x, p]
            new_mean = old_mean + (t - old_mean) / event_counts[y, x, p]
            mean_timestamps[y, x, p] = new_mean
            sum_of_squares[y, x, p] += (t - old_mean) * (t - new_mean)


        # Compute standard deviation
        std_dev_timestamps = np.sqrt(sum_of_squares / np.maximum(event_counts, 1))  # avoid division by zero

        ## TEMPORARY SOLUTION ##
        # Save the associated mask for this time window
        if mask_dict:  # Check if there are any masks
            closest_mask_timestamp = min(mask_dict.keys(), key=lambda x: abs(x - window_timestamp))  # Find the mask timestamp closest to the window timestamp
            print('closest mask timestamp', str(closest_mask_timestamp))
            
            my_mask = mask_dict[closest_mask_timestamp] / 1000
            my_mask = my_mask.astype(int)

            np.save(os.path.join(labels_dir, f'{scene_name}_mask_{window_index}.npy'), my_mask)
        else:
            print(f"No matching masks found for event windows in {dataset}")
            continue

        # Save the associated classical image for this time window
        if classical_dict:  # Check if there are any classical images
            closest_classical_timestamp = min(classical_dict.keys(), key=lambda x: abs(x - window_timestamp))  # Find the classical image timestamp closest to the window timestamp
            print('closest RGB timestamp', str(closest_classical_timestamp))
            np.save(os.path.join(images_dir, f'{scene_name}_classical_{window_index}.npy'), classical_dict[closest_classical_timestamp])
        else:
            print(f"No matching classical images found for event windows in {dataset}")
            continue

        # Concatenate processed data for this time window
        processed_data = np.concatenate([event_counts, mean_timestamps, std_dev_timestamps], axis=-1)
        # Save the processed data for this time window
        np.save(os.path.join(events_dir, f'{scene_name}_{window_index}.npy'), processed_data)

        window_index += 1

        if window_index == 5:
            break

if __name__ == "__main__":
    # Specify the input directories
    input_dir = '../datasets'
    # Specify the output directory
    output_dir = '../datasets/processed-fixed'

    # Constants
    width, height = int(640), int(480)
    duration_ms = 50.0

    print(f"Processing directory: {input_dir}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Event window duration: {duration_ms} ms")

    process_directory(input_dir, output_dir, width, height, duration_ms)

