import numpy as np
import os
import sys
import glob

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
    """
    Assumptions:
    The mask data in dataset_mask.npz are accessed using the filename specified in the gt_frame field of the frame info.
    A mask is associated with each event window by finding the mask whose timestamp is closest to the timestamp of the 
    last event in the window.
    The mask data are saved in the labels directory under output_path, with filenames formed by concatenating the scene 
    name, the string _mask_, and the window index. The mask data are multiplied by 1000 before saving.
    """
    # Load mask data
    mask_data = np.load(dataset + '_mask.npz')
    # Load dataset info
    dataset_info = np.load(dataset + '_info.npz', allow_pickle=True)['meta'].item()

    # Extract the frame info
    frame_infos = dataset_info['frames']

    # Create a dictionary to store the masks with their timestamps as keys
    #mask_dict = {frame_info['ts']: mask_data[f"mask_{frame_info['id']:010d}"] for frame_info in frame_infos if 'gt_frame' in frame_info}
    mask_dict = {}
    for frame_info in frame_infos:
        if 'gt_frame' in frame_info:
            mask_key = f"mask_{frame_info['id']:010d}"
            if mask_key in mask_data:
                mask_dict[frame_info['ts']] = mask_data[mask_key]

    # Split the path to get the scene name
    _, scene_name = os.path.split(os.path.dirname(dataset))

    # Create events directory if not exists
    events_dir = os.path.join(os.path.dirname(output_path), 'events')
    os.makedirs(events_dir, exist_ok=True)

    # Create labels directory if not exists
    labels_dir = os.path.join(os.path.dirname(output_path), 'labels')
    os.makedirs(labels_dir, exist_ok=True)

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

        # Save the associated mask for this time window
        ## TEMPORARY SOLUTION ##
        if mask_dict:  # Check if there are any masks
            closest_timestamp = min(mask_dict.keys(), key=lambda x: abs(x - window_timestamp))  # Find the mask timestamp closest to the window timestamp
            np.save(os.path.join(labels_dir, f'{scene_name}_mask_{window_index}.npy'), mask_dict[closest_timestamp] * 1000)
        else:
            print(f"No matching masks found for event windows in {dataset}")
            continue

        # Concatenate processed data for this time window
        processed_data = np.concatenate([event_counts, mean_timestamps, std_dev_timestamps], axis=-1)
        # Save the processed data for this time window
        np.save(os.path.join(events_dir, f'{scene_name}_{window_index}.npy'), processed_data)

        window_index += 1

        if window_index == 40:
            break

if __name__ == "__main__":

    # input_dir = sys.argv[1]
    # output_dir = sys.argv[2]
    input_dir = '../samsung_mono/imo'
    output_dir = '../samsung_mono/processed'

    # Constants
    width, height = int(640), int(480)
    duration_ms = 50.0

    print("processing", input_dir)
    print("width: "+ str(width) + "height: "+ str(height))
    print("duration: "+str(duration_ms))

    process_directory(input_dir, output_dir, width, height, duration_ms)