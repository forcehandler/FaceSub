import faceWarp
import cv2
import argparse
import sys
import skimage
from imutils.video import VideoStream

from imutils import face_utils
import dlib
import imutils

import skimage.transform
import numpy as np

from timeit import default_timer as timer

# Video file part
def warp_face_in_video(facial_mask_fn, video_in_fn, video_out_fn, show_video=False):
	"""
	Function to process frames in video file and 'replace' first found face by the the face from the still image.

	:param facial_mask_fn: path to the still image with a face
	:param video_in_fn: path to the input video file
	:param video_out_fn: path to the video file which will have 'replaced' face
	:param show_video: bool flag to show window with processed video frames
	"""

	facial_mask = cv2.imread(facial_mask_fn)
	facial_mask = cv2.cvtColor(facial_mask, cv2.COLOR_BGR2GRAY)
	facial_mask_lm = faceWarp.find_landmarks(facial_mask, faceWarp.predictor)

	video_in = cv2.VideoCapture(video_in_fn)

	video_out = cv2.VideoWriter(
		filename=video_out_fn,
		fourcc=cv2.cv.CV_FOURCC('m', 'p', '4', 'v'),
		frameSize=(int(video_in.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
				   int(video_in.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))),
		fps=25.0,
		isColor=True)

	total_frames_in = video_in.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
	while True:
		ret, frame_in = video_in.read()
		if ret == True:
			curr_frame = video_in.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
			frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
			if show_video:
				cv2.imshow('video_in', frame_in)
			else:
				print '{:.2%}\r'.format(curr_frame/total_frames_in),
				sys.stdout.flush()
			frame_out = faceWarp.face_warp(facial_mask, facial_mask_lm, frame_in)
			frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGR)
			video_out.write(frame_out)
			if show_video: cv2.imshow('video_out', frame_out)
			cv2.waitKey(1)
		else:
			video_in.release()
			video_in = None
			video_out.release()
			video_out = None
			cv2.destroyAllWindows()
			break

# Video cam part
def warp_face_from_webcam(facial_mask_fn, video_out_fn):
	"""
	Function to read video frames from the web cam, replace first found face by the face from the still image
	and show processed frames in a window. Also all processed frames will be save as a video.

	:param facial_mask_fn: path to the still image with a face
	:param video_out_fn: path to the video file which will have 'replaced' face
	"""

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("C:/Users/sonupc/Downloads/facial-landmarks/facial-landmarks/shape_predictor_68_face_landmarks.dat")

	facial_mask = cv2.cvtColor(cv2.imread(facial_mask_fn), cv2.COLOR_BGR2RGB)
	facial_mask_lm = faceWarp.find_landmarks(facial_mask, faceWarp.predictor)
	#print(facial_mask.shape)
	
	##cam = cv2.VideoCapture(0)
	cam = VideoStream().start()
	frame_size = (420, 240) # downsample size, without downsampling too many frames dropped

	################
	frame_in = cam.read()
	frame_in = cv2.resize(frame_in, dsize=frame_size)
	frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
	print(frame_in.shape , " FRAME IN")
	print(facial_mask.shape, 'facial mask')
	print(frame_in.shape[:2], 'face_warp dst_frame dimensions')
	print(skimage.img_as_ubyte(frame_in), "skimage transform")

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	##function unrolled code
	src_face, src_face_lm, dst_face = facial_mask, facial_mask_lm, frame_in
	output_shape = dst_face.shape[:3]  # dimensions of our final image (from webcam eg)

	# Get the landmarks/parts for the face.
	try:
		##dst_face_lm = find_landmarks(dst_face, predictor, opencv_facedetector=False)
		image, predictor, visualise, opencv_facedetector = dst_face, predictor, False, False
		start = timer()
		dets = detector(image, 1)
		end = timer()
		print(end - start, "dst_image detection time")
		try:
			start = timer()
			shape = predictor(image, dets[0])
			end = timer()
			print(end - start, "dst_image detection time")
			i = 0
			if visualise:
				while i < shape.num_parts:
					p = shape.part(i)
					cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), 2)
					i += 1
		except:
			shape = None
		dst_face_lm = shape
		# src_face_coord = _shape_to_array(src_face_lm)
		# dst_face_coord = _shape_to_array(dst_face_lm)
		src_face_coord = face_utils.shape_to_np(src_face_lm)
		dst_face_coord = face_utils.shape_to_np(dst_face_lm)
		warp_trans = skimage.transform.PiecewiseAffineTransform()
		warp_trans.estimate(dst_face_coord, src_face_coord)
		warped_face = skimage.transform.warp(src_face, warp_trans, output_shape=output_shape)
	except:
		warped_face = dst_face

	# Merge two images: new warped face and background of dst image
	# through using a mask (pixel level value is 0)
	##frame_out = _merge_images(warped_face, dst_face)
	img_top, img_bottom, mask = warped_face, dst_face, 0
	img_top = skimage.img_as_ubyte(img_top)
	img_bottom = skimage.img_as_ubyte(img_bottom)
	merge_layer = img_top == mask
	img_top[merge_layer] = img_bottom[merge_layer]
	frame_out = img_top
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	################
	# video_out = cv2.VideoWriter(
	# 	filename=video_out_fn,
	# 	fourcc=cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), # works good on OSX, for other OS maybe try other codecs
	# 	frameSize=frame_size,
	# 	fps=25.0,
	# 	isColor=True)

	while True:
		start = timer()
		frame_in = cam.read()
		# Downsample frame - otherwise processing is too slow
		#frame_in = cv2.resize(frame_in, dsize=frame_size)
		frame_in = imutils.resize(frame_in, width=400)

		frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)

		#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		# gray = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
		# # detect faces in the grayscale frame
		# rects = detector(gray, 0)
		# 	# loop over the face detections
		# for rect in rects:
		# 	# determine the facial landmarks for the face region, then
		# 	# convert the facial landmark (x, y)-coordinates to a NumPy
		# 	# array
		# 	shape = predictor(gray, rect)
		# 	shape = face_utils.shape_to_np(shape)
	 
		# 	# loop over the (x, y)-coordinates for the facial landmarks
		# 	# and draw them on the image
		# 	for (x, y) in shape:
		# 		cv2.circle(frame_in, (x, y), 1, (0, 0, 255), -1)
		  
		# # show the frame
		# cv2.imshow("Frame", frame_in)
		#key = cv2.waitKey(1) & 0xFF
		#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

		#frame_out = faceWarp.face_warp(facial_mask, facial_mask_lm, frame_in)
		#frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
		
		#video_out.write(frame_out)
		#faceWarp.draw_str(frame_out, (20, 20), 'ESC: stop recording  Space: stop & save video')

		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	##function unrolled code
		src_face, src_face_lm, dst_face = facial_mask, facial_mask_lm, frame_in
		output_shape = dst_face.shape[:3]  # dimensions of our final image (from webcam eg)

		# Get the landmarks/parts for the face.
		try:
			##dst_face_lm = find_landmarks(dst_face, predictor, opencv_facedetector=False)
			image, predictor, visualise, opencv_facedetector = dst_face, predictor, False, False
			dets = detector(image, 1)
			try:
				shape = predictor(image, dets[0])
				i = 0
				if visualise:
					while i < shape.num_parts:
						p = shape.part(i)
						cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), 2)
						i += 1
			except:
				shape = None
			dst_face_lm = shape
			# src_face_coord = _shape_to_array(src_face_lm)
			# dst_face_coord = _shape_to_array(dst_face_lm)
			src_face_coord = face_utils.shape_to_np(src_face_lm)
			dst_face_coord = face_utils.shape_to_np(dst_face_lm)
			warp_trans = skimage.transform.PiecewiseAffineTransform()
			warp_trans.estimate(dst_face_coord, src_face_coord)
			warped_face = skimage.transform.warp(src_face, warp_trans, output_shape=output_shape)
			
			
			
					
			
		except:
			warped_face = dst_face

		# Merge two images: new warped face and background of dst image
		# through using a mask (pixel level value is 0)
		##frame_out = _merge_images(warped_face, dst_face)
		img_top, img_bottom, mask = warped_face, dst_face, 0
		img_top = skimage.img_as_ubyte(img_top)
		img_bottom = skimage.img_as_ubyte(img_bottom)
		merge_layer = img_top == mask
		img_top[merge_layer] = img_bottom[merge_layer]
		frame_out = img_top
		frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		cv2.imshow('webcam', frame_out)

		end = timer()
		print("FPS: ", 1.0 / (end - start))

		ch = 0xFF & cv2.waitKey(1)
		if ch == 27:
			break
		if ch == ord(' '):
			break

	#cam.release()
	cv2.destroyAllWindows()
	cam.stop()

if __name__ == "__main__":
	# Let's parse running arguments and decide what we will do
	parser = argparse.ArgumentParser(description='Warp a still-image face around the other face in a video.')
	parser.add_argument('stillface', metavar='STILLFACE',
						help='the full path to jpg file with face', default='./face_mask.jpg')
	parser.add_argument('inputvideo', metavar='VIDEOIN',
						help='the full path to input video file where face will be changed. If "0" is provided \
						then the web cam will be used', default='0')
	parser.add_argument('outputvideo', metavar='VIDEOOUT',
						help='the full path to output video file with the new face. If "0" is provided then \
						process video will be shown on the screen, but not saved.')
	args = parser.parse_args()

	args.inputvideo = '0';
	args.stillface = './demo/terminator_crop.jpg';
	args.outputvideo = '0'
	# Check if there is a video file and process it.
	if args.inputvideo != '' and args.inputvideo != '0':
		try:
			print '*** Start processing for file: {} ***'.format(args.inputvideo)
			warp_face_in_video(args.stillface, args.inputvideo, args.outputvideo)
			print '\n*** Done! ***'
		except:
			print '*** Something went wrong. Error: {} ***'.format(sys.exc_info())

	# Otherwise use a webcam
	elif args.inputvideo == '0':
		try:
			print '*** Start webcam to save to file: {} ***'.format(args.outputvideo)
			warp_face_from_webcam(args.stillface, args.outputvideo)
			print '\n*** Done! ***'
		except:
			print '*** Something went wrong. Error: {} ***'.format(sys.exc_info())


