import argparse
import cv2
import datetime
import imutils
import mimetypes
import numpy as np
import os
import progressbar


from pathlib import Path
from skimage.metrics import structural_similarity
from subprocess import Popen, PIPE

def getMatchingTimestamps(input_file, ref_image):
    # Load the reference frame
    ref_frame = cv2.imread(ref_image)
    # Load the video file
    cap = cv2.VideoCapture(input_file)
    # Define the threshold for matching
    threshold = 0.8

    resized_ref = cv2.resize(ref_frame, (10, 10))
    resized_gray_ref = cv2.cvtColor(resized_ref, cv2.COLOR_BGR2GRAY)
    timestamp = -1
    matching_timestamps = []

    bar = progressbar.ProgressBar(maxval=20, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    # Loop through the frames of the video
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        # If there are no more frames, break the loop
        if not ret:
            break
        resized_frame = cv2.resize(frame, (10, 10))
        # Convert the reference frame and current frame to grayscale
        resized_gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        # Calculate the structural similarity between the reference frame and current frame
        (score, diff) = structural_similarity(resized_gray_ref, resized_gray_frame, full=True)

        # If the score is above the threshold, print the timestamp
        if score >= threshold:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        else:
            if (timestamp != -1):
                not_match_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                time_diff = not_match_timestamp - timestamp
                if (time_diff >= 1000):
                    matching_timestamps.append(timestamp)
                    timestamp = -1
    return matching_timestamps

def getMatchingTimestampsV2(input_file, ref_image):
    # Load the image to be searched for in the main image
    template_image = cv2.imread(ref_image)

    # Resize the template image
    template_image = cv2.resize(template_image, (0, 0), fx=0.5, fy=0.5)

    # Load the video file
    cap = cv2.VideoCapture(input_file)

    # Set the frame skip interval
    skip_frames = 10

    # Loop through the frames of the video
    frame_count = 0

    timestamp = -1
    matching_timestamps = []

    # Loop through the frames of the video
    while True:
        # Read a frame from the video
        ret, main_image = cap.read()
        # If there are no more frames, break the loop
        if not ret:
            break

        frame_count += skip_frames

        # Skip frames based on the skip interval
        if frame_count % skip_frames != 0:
            continue

        # Resize the main image
        main_image = cv2.resize(main_image, (0, 0), fx=0.5, fy=0.5)

        # Apply template matching using the cv2.TM_CCOEFF_NORMED method
        result = cv2.matchTemplate(main_image, template_image, cv2.TM_CCOEFF_NORMED)

        # Set a threshold for the match value
        threshold = 0.6

        # Find the locations of the matches in the result image
        locations = np.where(result >= threshold)

        if locations[0].size > 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        else:
            if (timestamp != -1):
                not_match_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                time_diff = not_match_timestamp - timestamp
                if (time_diff >= 1000):
                    matching_timestamps.append(timestamp)
                    timestamp = -1
    return matching_timestamps



            # Draw a rectangle around each location
            #for loc in zip(*locations[::-1]):
            #    cv2.rectangle(main_image, loc, (loc[0] + template_width, loc[1] + template_height), (0, 255, 0), 2)

            # Display the result
            #cv2.imshow('Result', main_image)
            #cv2.waitKey(0)

def getMatchingTimestampsV3(input_file, ref_image):
    # Load the image to be searched for in the main image
    template_image = cv2.imread(ref_image)

    # Resize the template image
    template_image = cv2.resize(template_image, (0, 0), fx=0.5, fy=0.5)

    # Load the video file
    cap = cv2.VideoCapture(input_file)

    # Get the duration of the video in milliseconds
    #total_duration = cap.get(cv2.CAP_PROP_POS_MSEC)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count/fps * 1000

    # Set the initial search window size
    search_window = 60 * 1000

    # Set the frame skip interval
    skip_frames = 10

    # Loop through the frames of the video
    frame_count = 0

    timestamp = -1
    #matching_timestamps = []

    # Set a threshold for the match value
    threshold = 0.6

    #print("Search window: ", str(search_window))
    #print("Total duration: ", str(total_duration))

    # Loop through the frames of the video until a match is found or we exceed half
    while search_window <= (total_duration / 2):

        start_time = (total_duration / 2) - (search_window / 2)
        end_time = (total_duration / 2) + (search_window / 2)

        #print("Start search at " + str(start_time) + " and end search at " + str(end_time) + " for a duration of " + str(search_window))

        # Set the capture position to the start time of the search window
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time)

        # Loop through the frames within the search window
        while cap.get(cv2.CAP_PROP_POS_MSEC) < end_time:
            # Read a frame from the video
            ret, main_image = cap.read()
            # If there are no more frames, break the loop
            if not ret:
                break

            # Skip frames based on the skip interval
            frame_count += skip_frames
            if frame_count % skip_frames != 0:
                continue

            # Resize the main image
            main_image = cv2.resize(main_image, (0, 0), fx=0.5, fy=0.5)

            # Apply template matching using the cv2.TM_CCOEFF_NORMED method
            result = cv2.matchTemplate(main_image, template_image, cv2.TM_CCOEFF_NORMED)

            # Find the locations of the matches in the result image
            locations = np.where(result >= threshold)

            # If we found at least one match
            if locations[0].size > 0:
                # Store the current timestamp (override previous timestamp if needed)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            else: # Else, if no match was found on this frame
                if (timestamp != -1): # But a previous one was found
                    # Calculate the elapsed time since the previously found timestamp
                    not_match_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                    time_diff = not_match_timestamp - timestamp
                    # If there was more than one second without a match, store the timestamp in the list of matches and reset the timestamp
                    if (time_diff >= 1000):
                        #matching_timestamps.append(timestamp)
                        return timestamp

        # Expand the search window by 1 minute
        print("No match found, expanding search window")
        search_window += search_window

def splitVideo(input, output, timestamp):
    if (output is None):
        dir_path, filename = os.path.split(input)
        dir_path += "/"   # Adding a trailing slash to the directory path
        filename_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(dir_path, (filename_no_ext + ".mkv"))
    # Convert milliseconds to a datetime object
    timestamp_dt = datetime.datetime.utcfromtimestamp(timestamp / 1000.0)
    # Format datetime object as string with format hh:mm:ss.ms
    timestamp_str = "timecodes:" + timestamp_dt.strftime('%H:%M:%S.%f')
    #print("Using timestamp : " + timestamp_str)
    print(f"MkvMerge cmd: mkvmerge -o {output_path} --split {timestamp_str} {input}")
    session = Popen(['mkvmerge', '-o', output_path, '--split', timestamp_str, input], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    res_text = session.communicate()
    res_text = res_text[0]

def get_video_files(folder_path):
    video_files = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_mime_type, _ = mimetypes.guess_type(file_name)
            if file_mime_type is not None and file_mime_type.startswith('video/'):
                video_files.append(os.path.join(folder_path, file_name))
    return video_files


if __name__ == "__main__":
    # Create an argument parser and define the expected arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input file or folder', required=True)
    parser.add_argument('-r', '--ref', type=str, help='Reference image', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output folder, default to same folder as input')

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Get the values of the named arguments
    input = args.input
    ref_image = args.ref
    output_folder = args.output

    if os.path.isdir(input):
        for file in get_video_files(input):
            print("Found file: ", str(file))
            timestamp = getMatchingTimestampsV3(file, ref_image)
            splitVideo(file, output_folder, timestamp)
    elif os.path.isfile(input):
        timestamp = getMatchingTimestampsV3(input, ref_image)
        splitVideo(input, output_folder, timestamp)
