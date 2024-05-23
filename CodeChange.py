



emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad","surprised", "unknown"]

# protoFile : to generate code that can read and write the data in the programming language of your choice
protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18

# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16]]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

pose_colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


def _pt_is_in(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box

    box_min_x = min(x1, x2)
    box_max_y = max(x1, x2)
    box_min_y = min(x1, x2)
    box_max_x = max(x1, x2)

    return box_max_x <= x <= box_max_x and box_min_y <= y <= box_max_y

# This function are returned as a list of tuples , where each tuple contain the coordinate(x,y) 
# and the score of the keypoints 

from collections import defaultdict
from copy import deepcopy
from turtle import right, width
import cv2 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import conftest
from sympy import fps
from ultralytics import YOLO

from writefile.neW.codework import _pt_is_in_box

def getKeypoints(probMap, threshold = 0.1):
    # Extract keypoints from a probability map using a threshold

    # Patameters :
    # probMap(np.ndarray): The probability map from which keypoints are extracted.
    # threshold(float) : The threshold value for detecting keypoints

    # Returns:
    # list: A list of keypoints, where each keypoint is represented as a tuple (x, y, score).
   

    # Apply Gaussian blur to smooth the probability map
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)

    # Create a binary mask where pixels greater than the threshold are set to 1
    mapMask = np.uint8(mapSmooth > threshold)

    keypoints = []

    # Find contours in the mask
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # For each contour, find the maxima
    for cnt in contours:
        # Create a mask for the current blob
        blobMask = np.zeros(mapMask.shape, dtype=np.uint8)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)

        # Apply the blob mask to the smoothed probability map
        maskedProbMap = mapSmooth * blobMask

        # Find the maximum value and its location in the masked probability map
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)

        # Append the keypints as tuple(x,y,scroe)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints  

def getValidPairs(output,frameWidth,frameHeight,detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



#This pleace need import numpy
import numpy as np

# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
# function (getPersonwiseKeypoints) aim to consolidate detected keypoints into a structure where each row corresponds to person and contains indicses of keypoints and a cumulative score
 
def getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    print("2",len(keypoints_list))
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

def compute_head_tall(head_size, total_height):

    # Constant for fine tuning the ratio
    FINE_TUNE_CONSTANT = 1.4

    # Calculate the ratio
    detect_ratio = (head_size * FINE_TUNE_CONSTANT / total_height)

    # Round the ratio to three decimal places
    detect_ratio = round(detect_ratio, 3)

    return detect_ratio


import numpy as np
import torch
from ultralytics.data.augment import LetterBox

def preprocess(img: np.ndarray, size int = 640) -> torch.Tensor:

    # Resize image to fit within a square of the specified size
    img = LetterBox(size, True)(image = img)

    # Recoder channels from HWC to CHW and convert from BGR to RGB
    img = img.transpose((2,0,1))[::-1]

    # Make array contiguous in memeory
    # numpy.ascontiguousarray()function is used when we want to return a contiguous array in memory (C order).
    img = np.ascontiguousarray(img)

    # Convert tensor to float (fp32)
    img = img.float()

    # Normalize pixel value to range [0,1]
    img /= 255

    # Add a batch dimension(1, C,H,W)
    return img.unsqueeze(0)



#  Create New Project
from xml.parsers.expat import model
import torch
from some_module import ops, colors # type: ignore

def postprocess(preds, img, orig_img, confthres, iouthres):
    preds = ops.non_max_suppression(preds,conf_thres=confthres,classes=[0],agnostic=False,max_det=100)

    for i, preds in enumerate(preds):
        shape = orig_img.shape

        # Scale boxes back to original image size
    return preds   

def draw_bbox(pred, names, annotator):
    # Draw bounding boxes with labels on the image

    for *xyxy, conf, cls in reversed(pred):

        # Convert class to integer
        c = int(cls)

        # Create label with class name and confidence
        label = f'{names[c]} {conf: 2f}'

        # Draw the bounding box with label
        annotator.box_label(xyxy, label, color= colors(c, True))

def main():
    # Placeholder : load your model, image, original image, class names, and annotator
    moddel = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    Image = torch.randn(1, 3, 640, 640)
    original_image = torch.randn(640, 640, 3)
    class_names = ['person']
    annotator = annotator(original_image)

    # Perform interence សន្និដ្ឋាន
    predictions = model(Image)

    # Define thresholds
    conf_threshold = 0.5
    iou_threshold = 0.4

    # Post-process the predictions
    processed_preds = postprocess(predictions, Image,original_image, conf_threshold, iou_threshold )

    # Draw bounding boxes on the original image
    draw_bbox(processed_preds, class_names, annotator)


if __name__ == "__main__":   
    main()

# Convert OpenCV image to PIL image 
import cv2 
from PIL import Image
import numpy as np 

def cv2pil(image):
    new_image = image.copy()
    # Gray scale
    if new_image.ndim == 2:
        pass

    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BAYER_BG2BGR)

    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BAYER_BG2BGR)    

    new_image = Image.fromarray(new_image)
    return new_image

# Example usage
if __name__ == "__main__":
    cv_image = cv2.imread('Project/image/5072846.png')

    # Convert the OpenCV image to PIL image
    pil_image = cv2pil(cv_image)

    # Display the PIL image
    pil_image.show()




# COnvert PIL image to OpenCv image
def pil2cv(image):

    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass

    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR.RGBA2BGRA)    

    return new_image
# Example usage

if __name__ == "__main__":

    cv_image = cv2.imread('Project/image/5072846.png') 
    pil_image = cv2pil(cv_image)
    pil_image.show()
    converted_cv_image = pil2cv(pil_image)

    # Display the OpenCV image in a window
    cv2.imshow('Converted Image', converted_cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution))) # type: ignore
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def expand_box(head_box,width,height,expand_size):
    expand_head_box = head_box
    expand_head_box[0] = head_box[0] - expand_size if head_box[0] > expand_size else 0
    expand_head_box[1] = head_box[1] - expand_size if head_box[1] > expand_size else 0
    expand_head_box[2] = head_box[2] + expand_size if width - head_box[2] > expand_size else width
    expand_head_box[3] = head_box[3] + expand_size if width - head_box[3] > expand_size else height
    return expand_head_box

def expand_box_ratio(head_box,width,height,expand_ratio):
    '''
    expand_ratio:元のhead_boxのwidth,heightのどれくらいを増やすか
    '''
    expand_head_box = head_box
    box_w = head_box[2]- head_box[0]
    box_h = head_box[3]- head_box[1]
    expand_size_w = int(box_w * expand_ratio)
    expand_size_h = int(box_h * expand_ratio)
    expand_head_box[0] = head_box[0] - expand_size_w if head_box[0] > expand_size_w else 0
    expand_head_box[1] = head_box[1] - expand_size_h if head_box[1] > expand_size_h else 0
    expand_head_box[2] = head_box[2] + expand_size_w if width - head_box[2] > expand_size_w else width
    expand_head_box[3] = head_box[3] + expand_size_h if width - head_box[3] > expand_size_h else height
    return expand_head_box



import numpy as np
import math

def get_angle(x0,y0,x1,y1,x2,y2):
    vec1 = [x1 - x0, y1 - y0]
    vec2 = [x2 - x0, y2 - y0]

    absvec1 = np.linalg.norm(vec1)
    absvec2 = np.linalg.norm(vec2)

    inner = np.inner(vec1, vec2)
    cos_theta = inner / (absvec1 * absvec2)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.degrees(math.acos(cos_theta))

    return theta



import cv2
class Opt:
    def __init__(self):
        self.output = 'output.mp4'
        self.source = 0 
        self.confthres = 0.5
        self.iouthres = 0.5
        self.device = 'cpu'
        self.gaze = False
        self.person_count = False
        self.to_csv = False
        self.db = False

def detect(frame, opt):
    # Example detect function logic : Add some text to the frame
    out, source, confthres, iouthres, device = opt.output, opt.source, opt.confthres, opt.iouthres, opt.device

    is_gaze = opt.gaze
    is_count_person = opt.person_count
    is_to_csv = opt.to_csv
    is_db = opt.db

    if is_to_csv == "True":
        is_gaze = "True"
        is_count_person = "True"

    # Example processing : Dram a rectangle and put text
    cv2.rectangle(frame, (50, 50), (200, 200),(0, 255, 0, 2))
    cv2.putText(frame, 'Processed Frame', (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def main():
    # Initialize options
    opt = Opt()

    # Open the video source
    video_capture = cv2.VideoCapture(opt.source)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

# File format and VideoWriter initialization
# File format (MP4)

fmt = cv2.VideoWriter.fourcc('m','p','4','v')
writer = cv2.VideoWriter(Opt.output, fmt, 10,(width, right))

frame_number = 1
time_fromStart = 0

print(width, height, all_frame_count, fps) # type: ignore

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read() # type: ignore


    if not ret:
        print("End of video or cannot read the frame")
        break

    # Process the frame using detect function
    processed_frame = detect(frame, opt) # type: ignore
    writer.write(processed_frame)

    # Display the processed frame
    cv2.imshow('Processed Video', processed_frame)

    # Exot if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1


#　torch.cuda.set_device cannot use to set cpu device, but give an ambiguity hint
# GPU check
	# 
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device check")
    print(torch.device)

    # for person_detection=========================================================
    person_model = YOLO("best.pt")
    # model = YOLO("mymodel/yolov8n-pose.pt")
    if torch.device == "gpu":
        person_model.to('cuda:0')
    person_color = (0, 255, 0)
    
    # for openpose=========================================================
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    if torch.device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif torch.device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device") 


    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight/height)*width) # type: ignore
    threshold = 0.1

    CHILD_THRESHOLD = 0.18

    # Gaze estimation is the process of determining where a person 
    # is looking based on their eye movements and facial features.

# for gaze==============================================================
    if is_gaze == "True":     # type: ignore
        # チューニングしたやつ
        face_model = YOLO("mymodel/yolov8face2.pt")
        if torch.device == "gpu":
            face_model.to('cuda:0')
        # set up data transformation
        test_transforms = _get_transform()

        model = ModelSpatial() # type: ignore
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_weights, map_location=torch.device('cpu') )
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if torch.device == "gpu":
            model.cuda()
        model.train(False)

        # 視野角は150ど
        rotation_angle = 75
        # そのなかで注目してるであろう30ど
        rotation_angle_default = 15
        default_diameter = 200
    face_color = (200,0,0)

# for db : DB is a file is organnized for storing data

import librosa
import numpy as np
from moviepy.editor import AudioFileClip
from transformers import pipeline # type: ignore
from some_emotion_recognition_module import cut_wav, load_model, audio_from_path, transcribe, write_wav_result  # type: ignore # Assuming these functions are defined elsewhere

def process_audio(source, is_db=True):
    if is_db:
        # Generate AudioFileClip object from video
        audio = AudioFileClip(source)
        wav_path = source.replace('.mp4', '.wav')
        # Save as .wav file
        audio.write_audiofile(wav_path)
        y, sr = librosa.load(wav_path)

        # RMS : Root Mean Square
        # Calculate RMS
        rms = librosa.feature.rms(y=y)
        # Convert RMS to dB
        db = librosa.amplitude_to_db(rms)
        time = librosa.times_like(db, sr=sr)
        
        time_from_start_db = 0
        db_at_this_second = 0
        db_max_at_this_second = -100
        time_num = 0
        db_per_second = []
        db_max_per_second = []

        for i, t in enumerate(time):
            if time_from_start_db <= t < time_from_start_db + 1:
                db_at_this_second += db[0][i]
                db_max_at_this_second = max(db_max_at_this_second, db[0][i])
                time_num += 1
            else:
                db_ave_at_this_second = db_at_this_second / time_num if time_num > 0 else -100
                db_per_second.append(db_ave_at_this_second)
                db_max_per_second.append(db_max_at_this_second)
                db_at_this_second = db[0][i]
                db_max_at_this_second = db[0][i]
                time_num = 1
                time_from_start_db += 1


        # Emotion Analysis
        wave_file_list = cut_wav(wav_path, 5)
        emotion_labels = ["happy", "sad", "angry", "neutral"]  # Define your emotion labels here
        emotion_list = []
        emotion_per_second = {label: [] for label in emotion_labels}

        inference_pipeline = pipeline(
            task="emotion-recognition",
            model="iic/emotion2vec_base_finetuned",
            model_revision="v2.0.4"
        )

        for cut_wav_path in wave_file_list:
            rec_result = inference_pipeline(cut_wav_path, output_dir="./output", granularity="utterance", extract_embedding=False)
            emotion_list.append(rec_result)
            scores = rec_result[0]["scores"]
            for key, value in zip(emotion_labels, scores):
                emotion_per_second[key].append(value)

        # Transcription
        trans_model = load_model()
        audio = audio_from_path(wav_path)
        ret = transcribe(trans_model, audio)
        
        # Write results
        write_wav_result(ret, emotion_list, "./output_audio_result")  # Adjust parameters as needed

# Example usage
    source_path = "path_to_your_video.mp4"
    process_audio(source_path)




# Carefully Points
# If "person being watched time" refers to tracking how long a specific person appears in the video
    personbeingWatchedTime = defaultdict(int)

    personbeingWTime = defaultdict(int)

    degreeofAttentionatSecond = []

    degreeofAttentionatthissec = 0

    id_attention = defaultdict(list)

    id_attention_at_this_time = defaultdict(int)

    faceWatchingTime = defaultdict(int)
    facebeingTime = defaultdict(int)

    id_exist = defaultdict(list)
    

    id_exist_at_this_time = defaultdict(int)
     
    personatSecond = []
 
    personcountatthissec = 0

    personIsChild = {}
    idBox = {}

    frame = 0


# Process video frame by frame

    while video_capture.isOpened(): # type: ignore
        # Load frames one by one
        ret, img = video_capture.read() # type: ignore

        if frame_number % fps < 1:
            id_attention[time_fromStart] = id_attention_at_this_time
            id_exist[time_fromStart] = id_exist_at_this_time
            time_fromStart += 1
        #   Can a person with a certain ID be seen in a certain second? Initialize every second, add every second
            id_attention_at_this_time = defaultdict(int)
        #   Is there a person with a certain ID in a certain second? Initialize every second, add every second
            id_exist_at_this_time = defaultdict(int)

        # Exit the loop when the video finishes playing
        if not ret:
            break
        frame += 1

        origin = deepcopy(img)
        output_img = deepcopy(img)

        # track person===============================================================================
        # resuresults_personlts = person_model.track(img,tracker = "bytetrack.yaml",persist=True,conf=confthres,iou=iouthres,classes=[0])
        results_person = person_model.track(img,persist=True,conf=conftest,iou=iouthres,classes=[0]) # type: ignore

        if is_count_person == "True": # type: ignore
            personcountatthissec += len(results_person[0])
            if frame_number % fps < 1:
                avePersonCountatSecond = int(personcountatthissec / fps)
                personatSecond.append(avePersonCountatSecond)
                personcountatthissec = 0
            person_count = "person_count: " + str(len(results_person[0]))
            cv2.putText(output_img, person_count, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)
        else: # count_personをしない時
            pass

        if is_gaze == "True": # type: ignore
            
        # track gaze===============================================================================
            results_face = face_model.track(img,persist=True,conf=confthres,iou=iouthres) # type: ignore
        # # Check if there are any detections
        # if results_face[0].boxes is not None:

            # Center coordinates of where everyone is looking
            gaze_list = []

            with torch.no_grad():
                frame_raw = cv2pil(output_img)
                frame_raw = frame_raw.convert('RGB')
                width, height = frame_raw.size
                print(width,height)
                # Extract IDs if they exist
                ids = results_face[0].boxes.id.cpu().numpy().astype(int) if results_face[0].boxes.id is not None else []

                # Annotate frame with boxes and IDs
                for i, box in enumerate(results_face[0].boxes.xyxy.cpu().numpy().astype(int)):
                    id = ids[i] if len(ids)!=0 else None
                    head_box = [int(box[0]),int(box[1]),int(box[2]),int(box[3])]
                    
                    is_in_adult = False
                    head_pt1 = (head_box[0], head_box[1])
                    head_pt2 = (head_box[2], head_box[3])
                    for p_id, p_box in idBox.items():
                        expand_p_box = expand_box_ratio(p_box, width, height, 0.1)
                        if not personIsChild[p_id] and _pt_is_in_box(head_pt1, expand_p_box) and _pt_is_in_box(head_pt2, expand_p_box):
                            is_in_adult = True
                            break
                    
                    if not is_in_adult:
                        continue
                    
                    head_box = expand_box_ratio(head_box,width,height,0.6)
                    head = frame_raw.crop((head_box)) # head crop

                    head = test_transforms(head) # transform inputs
                    frame = test_transforms(frame_raw)
                    head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height, resolution=input_resolution).unsqueeze(0) # type: ignore


                    head = head.unsqueeze(0)
                    frame = frame.unsqueeze(0)
                    head_channel = head_channel.unsqueeze(0)

                # CUDA Python. CUDA® Python provides Cython/Python wrappers for CUDA driver and runtime APIs;
                # detch() : open source deep learning

                    if device == "gpu": # type: ignore
                        head = head.cuda()
                        frame = frame.cuda()
                        head_channel = head_channel.cuda()

                    # forward pass
                    raw_hm, _, inout = model(frame, head_channel, head)

                    # heatmap modulation
                    raw_hm = raw_hm.cpu().detach().numpy() * 255
                    raw_hm = raw_hm.squeeze()
                    inout = inout.cpu().detach().numpy()
                    inout = 1 / (1 + np.exp(-inout))
                    inout = (1 - inout) * 255

                    norm_map= np.array(Image.fromarray(obj=raw_hm, mode='F').resize(size=(width, height), resample=Image.BICUBIC))

                    gray_image = np.zeros((norm_map.shape[0], norm_map.shape[1]), dtype=np.uint8)
                    cv2.normalize(norm_map, gray_image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    heatmap = gray_image

                # argmax is a function which gives the index of the greatest number in the given row or column
                    max_index = np.argmax(heatmap)
                    h = 1

                    max_index += 1
                    while max_index - width > 0:
                        max_index = max_index - width
                        h += 1
                    
                    # center of sight
                    h_point = (1.0 / (height -1)) * (h-1)
                    w_point = (1.0 / (width -1)) * (max_index-1)

                    gaze_center_h = h_point*height
                    gaze_center_w = w_point*width
                    head_w = int((head_box[2]+head_box[0])/2)
                    head_h = int((head_box[3]+head_box[1])/2)
                    gaze_list.append((head_w,head_h,gaze_center_h,gaze_center_w))

                    # ヒートマップ表示するならこれ---------------
                    # カラーマップを適用
                    # heatmap_color = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
                    # threshold,alpha = 200,0.7
                    # mask = np.where(heatmap<=threshold, 1, 0)
                    # mask = np.reshape(mask, (height, width, 1))
                    # mask = np.repeat(mask, 3, axis=2)
                    # marge = output_img*mask + heatmap_color*(1-mask)
                    # marge = marge.astype("uint8")   
                    # output_img = cv2.addWeighted(output_img, 1-alpha, marge,alpha,0)                      

                    original_output_image = output_img.copy()

                    center_h = height * h_point
                    center_w = width * w_point
                    diameter = int(np.sqrt(abs(center_h - head_h) ** 2 + abs(center_w - head_w)** 2))+40
                    # Get radian units
                    radian = math.atan2((center_h - head_h),(center_w - head_w))
                    # Get angle from radians
                    degree = radian * (180 / math.pi)

                    # 人間の視野である150度をプロット
                    cv2.ellipse(output_img, (head_w, head_h), (diameter*10, diameter*10), degree,-rotation_angle, rotation_angle, face_color, thickness=-1)
                    output_img = cv2.addWeighted(original_output_image, 0.8, output_img, 0.2, 0)
                    # 距離が200は扇形の角度は15*2=30度にする
                    _rotation_angle = int(rotation_angle_default * default_diameter / diameter)
                    cv2.ellipse(output_img, (head_w, head_h), (diameter*10, diameter*10), degree,-_rotation_angle, _rotation_angle, face_color, thickness=-1)
                    output_img = cv2.addWeighted(original_output_image, 0.5, output_img, 0.5, 0)

                    # 顔の描画
                    cv2.rectangle(output_img, (head_box[0],head_box[1]), (head_box[2],head_box[3]), face_color, 2, cv2.LINE_AA)

                    if id is not None:
                        # Calculate the total time that appears on the screen
                        facebeingTime[id] += 1 
                        # 人をどれくらい見てるか
                        if results_person[0].boxes is not None:
                            for i, box in enumerate(results_person[0].boxes.xyxy.cpu().numpy().astype(int)):
                                boxx_center = int((box[2]+box[0])/2)
                                boxy_center = int((box[3]+box[1])/2)

                                theta = get_angle(head_w,head_h,gaze_center_w,gaze_center_h,boxx_center,boxy_center)
                                if theta <= rotation_angle:
                                # if ((box[0] - 50) < gaze_center_w < (box[2] + 50)) and ((box[1] - 50) < gaze_center_h < (box[3] + 50)) and head_box[1] < box[1]:
                                    faceWatchingTime[id] += 1
                                    break
                            degreeofWatch = int((faceWatchingTime[id]/facebeingTime[id])*100)

                        # cv2.putText(output_img, f"ID {id}", (head_box[0], head_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, face_color, 2)
                        cv2.putText(output_img, f"faceID {id} Watch {degreeofWatch}", (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,face_color, 2)

             # draw_person depend on gaze--------------------------------------------------------------------------
            # openpose--------------------------------------------------------------------------
            inpBlob = cv2.dnn.blobFromImage(origin, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()
            detected_keypoints = []
            keypoints_list = np.zeros((0,3))
            keypoint_id = 0
            threshold = 0.1

            for part in range(nPoints):
                probMap = output[0,part,:,:]
                probMap = cv2.resize(probMap, (origin.shape[1], origin.shape[0]))
                keypoints = getKeypoints(probMap, threshold)
                # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
                keypoints_with_id = []
                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(output_img, detected_keypoints[i][j][0:2], 5, pose_colors[i], -1, cv2.LINE_AA)
            # cv2.imshow("Keypoints",frameClone)
            valid_pairs, invalid_pairs = getValidPairs(output,width,height,detected_keypoints)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list)

            # Check if there are any detections
            if results_person[0].boxes is not None:
                
                # Extract IDs if they exist
                ids = results_person[0].boxes.id.cpu().numpy().astype(int) if results_person[0].boxes.id is not None else []

                # Annotate frame with boxes and IDs
                for i, box in enumerate(results_person[0].boxes.xyxy.cpu().numpy().astype(int)):
                    id = ids[i] if len(ids)!=0 else None
                    this_person_color = person_color

                    # 画面内に現れる総時間を計算
                    if id is not None:  
                        personbeingWTime[id] += 1    
                        id_exist_at_this_time[id] = 1   
                        if id not in personIsChild.keys():
                            personIsChild[id] = False
                        
                        for n in range(len(personwiseKeypoints)):
                            if personwiseKeypoints[n][1] < 0 or personwiseKeypoints[n][10] < 0 or personwiseKeypoints[n][16] < 0 or personwiseKeypoints[n][17] < 0:
                                continue

                            neck = keypoints_list[int(personwiseKeypoints[n][1])]
                            # rhip = detected_keypoints[8]    
                            # rknee = detected_keypoints[9]
                            rankle = keypoints_list[int(personwiseKeypoints[n][10])]
                            r_ear = keypoints_list[int(personwiseKeypoints[n][16])]
                            l_ear = keypoints_list[int(personwiseKeypoints[n][17])]
                            
                            if _pt_is_in_box(neck, box) and _pt_is_in_box(rankle, box) and _pt_is_in_box(r_ear, box) and _pt_is_in_box(l_ear, box):
                                                                                            
                                head = l_ear[0] - r_ear[0]
                                p_height = (rankle[1] - neck[1]) + head
                                detect_ratio= compute_head_tall(head, p_height)
                                if detect_ratio > CHILD_THRESHOLD:
                                    personIsChild[id] = True                        
                        
                    # みられている場合は...色の変更，見られている時間を加算
                    for (face_x,face_y,center_h,center_w) in gaze_list:
                        # print(center_w,box[0],box[2])
                        boxx_center = int((box[2]+box[0])/2)
                        boxy_center = int((box[3]+box[1])/2)
                        theta = get_angle(face_x,face_y,center_w,center_h,boxx_center,boxy_center)

                        if theta <= rotation_angle:
                        # if ((box[0] - 50) < center_w < (box[2] + 50)) and ((box[1] - 50) < center_h < (box[3] + 50)) and face_y < box[1]:
                            this_person_color = (180,180,180)
                            if id is not None:
                                personbeingWatchedTime[id] += 1
                                id_attention_at_this_time[id] = 1
                            break
                    cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), this_person_color, 2)
                    if id is not None:
                        degreeofAttention = int((personbeingWatchedTime[id]/personbeingWTime[id])*100)
                        degreeofAttentionatthissec += degreeofAttention
                        
                        idBox[id] = box

                        if personIsChild[id]: # 元々child⇒person
                            cv2.putText(output_img, f"ID {id}: child  ATTENTION {degreeofAttention}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,this_person_color, 2)
                        else: # 元々adult⇒person
                            cv2.putText(output_img, f"ID {id}: adult", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,this_person_color, 2)

            # 一秒あたりの注目度の合計を求める
            if frame_number % fps < 1:
                aveDegreeofAttentionatSecond = int(degreeofAttentionatthissec / fps)
                degreeofAttentionatSecond.append(aveDegreeofAttentionatSecond)
                degreeofAttentionatthissec = 0
            
            
            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(output_img, (B[0], A[0]), (B[1], A[1]), pose_colors[i], 3, cv2.LINE_AA)

        elif is_gaze == "False": # gaze情報を表示しない時
            # draw_person depend on gaze--------------------------------------------------------------------------
            
            # Check if there are any detections
            if results_person[0].boxes is not None:
                # Extract IDs if they exist
                ids = results_person[0].boxes.id.cpu().numpy().astype(int) if results_person[0].boxes.id is not None else []

                # Annotate frame with boxes and IDs
                for i, box in enumerate(results_person[0].boxes.xyxy.cpu().numpy().astype(int)):
                    id = ids[i] if len(ids)!=0 else None
                    cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), person_color, 2)
                    if id is not None:
                        cv2.putText(output_img, f"ID {id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, person_color, 2)

                   # draw_person depend on gaze--------------------------------------------------------------------------
            # openpose--------------------------------------------------------------------------
            inpBlob = cv2.dnn.blobFromImage(origin, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()
            detected_keypoints = []
            keypoints_list = np.zeros((0,3))
            keypoint_id = 0
            threshold = 0.1

            for part in range(nPoints):
                probMap = output[0,part,:,:]
                probMap = cv2.resize(probMap, (origin.shape[1], origin.shape[0]))
                keypoints = getKeypoints(probMap, threshold)
                # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
                keypoints_with_id = []
                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(output_img, detected_keypoints[i][j][0:2], 5, pose_colors[i], -1, cv2.LINE_AA)
            # cv2.imshow("Keypoints",frameClone)
            valid_pairs, invalid_pairs = getValidPairs(output,width,height,detected_keypoints)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list)

            # Check if there are any detections
            if results_person[0].boxes is not None:
                
                # Extract IDs if they exist
                ids = results_person[0].boxes.id.cpu().numpy().astype(int) if results_person[0].boxes.id is not None else []

                # Annotate frame with boxes and IDs
                for i, box in enumerate(results_person[0].boxes.xyxy.cpu().numpy().astype(int)):
                    id = ids[i] if len(ids)!=0 else None
                    this_person_color = person_color

                    # 画面内に現れる総時間を計算
                    if id is not None:  
                        personbeingWTime[id] += 1    
                        id_exist_at_this_time[id] = 1   
                        if id not in personIsChild.keys():
                            personIsChild[id] = False
                        
                        for n in range(len(personwiseKeypoints)):
                            if personwiseKeypoints[n][1] < 0 or personwiseKeypoints[n][10] < 0 or personwiseKeypoints[n][16] < 0 or personwiseKeypoints[n][17] < 0:
                                continue

                            neck = keypoints_list[int(personwiseKeypoints[n][1])]
                            # rhip = detected_keypoints[8]    
                            # rknee = detected_keypoints[9]
                            rankle = keypoints_list[int(personwiseKeypoints[n][10])]
                            r_ear = keypoints_list[int(personwiseKeypoints[n][16])]
                            l_ear = keypoints_list[int(personwiseKeypoints[n][17])]
                            
                            if _pt_is_in_box(neck, box) and _pt_is_in_box(rankle, box) and _pt_is_in_box(r_ear, box) and _pt_is_in_box(l_ear, box):
                                                                                            
                                head = l_ear[0] - r_ear[0]
                                p_height = (rankle[1] - neck[1]) + head
                                detect_ratio= compute_head_tall(head, p_height)
                                if detect_ratio > CHILD_THRESHOLD:
                                    personIsChild[id] = True                        
                        
                    # みられている場合は...色の変更，見られている時間を加算
                    for (face_x,face_y,center_h,center_w) in gaze_list:
                        # print(center_w,box[0],box[2])
                        boxx_center = int((box[2]+box[0])/2)
                        boxy_center = int((box[3]+box[1])/2)
                        theta = get_angle(face_x,face_y,center_w,center_h,boxx_center,boxy_center)

                        if theta <= rotation_angle:
                        # if ((box[0] - 50) < center_w < (box[2] + 50)) and ((box[1] - 50) < center_h < (box[3] + 50)) and face_y < box[1]:
                            this_person_color = (180,180,180)
                            if id is not None:
                                personbeingWatchedTime[id] += 1
                                id_attention_at_this_time[id] = 1
                            break
                    cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), this_person_color, 2)
                    if id is not None:
                        degreeofAttention = int((personbeingWatchedTime[id]/personbeingWTime[id])*100)
                        degreeofAttentionatthissec += degreeofAttention
                        
                        idBox[id] = box

                        if personIsChild[id]: # 元々child⇒person
                            cv2.putText(output_img, f"ID {id}: child  ATTENTION {degreeofAttention}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,this_person_color, 2)
                        else: # 元々adult⇒person
                            cv2.putText(output_img, f"ID {id}: adult", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,this_person_color, 2)

            # Calculate the total attention level per second
            if frame_number % fps < 1:
                aveDegreeofAttentionatSecond = int(degreeofAttentionatthissec / fps)
                degreeofAttentionatSecond.append(aveDegreeofAttentionatSecond)
                degreeofAttentionatthissec = 0
            
            
            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(output_img, (B[0], A[0]), (B[1], A[1]), pose_colors[i], 3, cv2.LINE_AA)

        elif is_gaze == "False": # type: ignore # gaze情報を表示しない時
            # draw_person depend on gaze--------------------------------------------------------------------------
            
            # Check if there are any detections
            if results_person[0].boxes is not None:
                # Extract IDs if they exist
                ids = results_person[0].boxes.id.cpu().numpy().astype(int) if results_person[0].boxes.id is not None else []

                # Annotate frame with boxes and IDs
                for i, box in enumerate(results_person[0].boxes.xyxy.cpu().numpy().astype(int)):
                    id = ids[i] if len(ids)!=0 else None
                    cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), person_color, 2)
                    if id is not None:
                        cv2.putText(output_img, f"ID {id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, person_color, 2)

        else : # とりあえずgaze情報を表示しない時
            # draw_person depend on gaze--------------------------------------------------------------------------
            
            # Check if there are any detections
            if results_person[0].boxes is not None:
                # Extract IDs if they exist
                ids = results_person[0].boxes.id.cpu().numpy().astype(int) if results_person[0].boxes.id is not None else []

                # Annotate frame with boxes and IDs
                for i, box in enumerate(results_person[0].boxes.xyxy.cpu().numpy().astype(int)):
                    id = ids[i] if len(ids)!=0 else None
                    cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), person_color, 2)
                    if id is not None:
                        cv2.putText(output_img, f"ID {id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,person_color, 2)



        # OpenCVで表示＆キー入力チェック
        # cv2.imshow("YOLOv8 Tracking", output_img)
        writer.write(output_img)

        # 'q'キーが押された場合、ループを抜ける
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        frame_number += 1


    # Release used resources
    #video_capture.release()
    #writer.release()
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
    #cv2.waitKey(1)

    # if is_count_person == "True":    
    # PLot the data on a graph

    #   plt.plot(personatSecond)
    #   The title's graph
    #    plt.title('person count at sec')
    #    X and Y axis labels
    #     plt.xlabel('time from start')
    #     plt.ylabel('the number of person')
    #     グラフを表示
    #     plt.show()

    # デシベルのcsvと図を保存

    import pandas as pd
    import matplotlib.pyplot as plt

    if is_db == "True":   
        data = {'max_db': db_maxPerSecond, # type: ignore
                'ave_db': db_perSecond, # type: ignore
                }
        df = pd.DataFrame(data)
        df_t = df.T
        df_t.to_csv(opt.output_db_csv, index=True)  # type: ignore

    # PLot dB data over time 
        plt.xlabel("Time(s)")
        plt.ylabel("dB")
        plt.plot(time,db[0])
        plt.savefig(opt.output_db_png)   # type: ignore
        plt.close()

        df = pd.DataFrame(emotion_per_second)
        df_t = df.T
        df_t.to_csv(opt.output_emotion_csv, index=True)  # type: ignore


import pandas as pd

if is_to_csv == "True":  # type: ignore

    def convert_defaultdict_to_dataframe(data_dict, prefix):
        """
        Convert a defaultdict to a regular dictionary, then to a DataFrame,
        sort it, reindex it, and set custom row labels.
        """
        
        regular_dict = {key: dict(inner_dict) for key, inner_dict in data_dict.items()}
        df = pd.DataFrame(regular_dict).fillna(0).astype(int)

        # Sort columns
        df = df.reindex(sorted(df.columns), axis=1)

        # Sort index and fill missing indices with 0
        df = df.reindex(range(df.index.min(), df.index.max() + 1), fill_value=0)

        # Set custom row labels
        df_index = [f"{prefix}_{i}" for i in range(len(df))]
        df = df.set_axis(df_index, axis=0)

        return df


    # Process id_exist data
    df_id_exist = convert_defaultdict_to_dataframe(id_exist, "person") # type: ignore
    person_id_num = len(df_id_exist)
    df_id_exist.to_csv(opt.output_personid_csv, index=True)  # type: ignore

    # Process id_attention data
    df_id_attention = convert_defaultdict_to_dataframe(id_attention, "注目度_ID") # type: ignore

    # Ensure id_attention DataFrame has the same number of rows as id_exist DataFrame
    while len(df_id_attention) < person_id_num:
        df_id_attention.loc[f"注目度_ID_{len(df_id_attention)}"] = 0

    # Write id_attention DataFrame to CSV
    df_id_attention.to_csv(opt.output_attentionid_csv, index=True)  # type: ignore

    # Process person count data
    df_person_count = pd.DataFrame({'person_count': personatSecond}).T # type: ignore
    df_person_count.to_csv(opt.output_personcount_csv, index=True)  # type: ignore

  

import argparse
import time

def main():
    t1 = time.localtime()
    start_time = time.strftime("%H:%M:%S", t1)
    
    parser = argparse.ArgumentParser(description="Video processing with optional gaze detection and CSV output.")
    
    parser.add_argument('--source', type=str, default="data/movie_test.mp4", help='Input video source path')
    parser.add_argument('--output', type=str, default="output/test.mp4", help='Output video path')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size in pixels')
    parser.add_argument('--confthres', type=float, default=0.15, help='Object confidence threshold')
    parser.add_argument('--iouthres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--model_weights', type=str, default='mymodel/gaze/model_demo.pt', help='Path to model weights')
    
    # Output paths for various results
    parser.add_argument('--output_db_csv', type=str, default="output/db.csv", help='Output path for DB CSV')
    parser.add_argument('--output_emotion_csv', type=str, default="output/emotion.csv", help='Output path for emotion CSV')
    parser.add_argument('--output_db_png', type=str, default="output/db.png", help='Output path for DB PNG')
    parser.add_argument('--output_audio_result', type=str, default="output/audio_result.mp4", help='Output path for audio result')
    parser.add_argument('--output_personid_csv', type=str, default="output/person_id.csv", help='Output path for person ID CSV')
    parser.add_argument('--output_attentionid_csv', type=str, default="output/attention_id.csv", help='Output path for attention ID CSV')
    parser.add_argument('--output_personcount_csv', type=str, default="output/personcount.csv", help='Output path for person count CSV')
    
    # Flags for enabling features
    parser.add_argument('--gaze', type=str, default='True', help='Enable gaze information (True/False)')
    parser.add_argument('--person_count', type=str, default='True', help='Enable person-count information (True/False)')
    parser.add_argument('--to_csv', type=str, default='True', help='Enable CSV output (True/False)')
    parser.add_argument('--db', type=str, default='False', help='Enable DB CSV and graph output (True/False)')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run inference on (cpu/gpu)')
    
    args = parser.parse_args()
    detect(args)
    
    t2 = time.localtime()
    end_time = time.strftime("%H:%M:%S", t2)
    print("Start time:", start_time)
    print("End time:", end_time)

if __name__ == '__main__':
    main()



