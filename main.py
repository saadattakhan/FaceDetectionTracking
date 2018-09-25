from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

import mxnet as mx
from mtcnn_detector import MtcnnDetector

import dlib

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", required=True,
		help="path to input video file")
	args = vars(ap.parse_args())
	

	detector = MtcnnDetector(model_folder="model", ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)


	rectangleColor = (0,165,255)
	faceTrackers = {}
	faceScores = {}


	# start the file video stream thread and allow the buffer to
	# start to fill
	print("[INFO] starting video file thread...")
	fvs = FileVideoStream(args["video"]).start()
	time.sleep(1.0)
	 
	# start the FPS timer
	fps = FPS().start()
	# loop over frames from the video file stream

	frame_count=0
	frame_break=12
	face_ids=None
	while fvs.more():
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale (while still retaining 3
		# channels)
		frame = fvs.read()
		frame = imutils.resize(frame, width=1280)
		frame_count+=1

		process_frame = imutils.resize(frame, width=800)

		fidsToDelete = []
		for fid in faceTrackers.keys():
			trackingQuality = faceTrackers[ fid ].update( process_frame )

			#If the tracking quality is good enough, we must delete
			#this tracker
			if trackingQuality < 7:
				fidsToDelete.append( fid )

		for fid in fidsToDelete:
			print("Removing fid " + str(fid) + " from list of trackers")
			faceTrackers.pop( fid , None )



		# process_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# process_frame = np.dstack([frame, frame, frame])

		if frame_count%frame_break==0:
			results = detector.detect_face(process_frame)

			if results is not None:
				total_boxes = results[0]
				for box in total_boxes:
					left=int(box[0])-10
					top=int(box[1])-10
					right=int(box[2])+10
					bottom=int(box[3])+10
					confidence=box[4]

					left= 0 if left<0 else left
					top= 0 if top<0 else top
					right= process_frame.shape[1] if right>process_frame.shape[1] else right
					bottom= process_frame.shape[0] if bottom>process_frame.shape[0] else bottom
					if confidence>0.90:
						w = right-left
						h = bottom-top


						x_bar = left + 0.5 * w
						y_bar = top + 0.5 * h

						matchedFid = None

						for fid in faceTrackers.keys():
							tracked_position =  faceTrackers[fid].get_position()

							t_x = int(tracked_position.left())
							t_y = int(tracked_position.top())
							t_w = int(tracked_position.width())
							t_h = int(tracked_position.height())


							t_x_bar = t_x + 0.5 * t_w
							t_y_bar = t_y + 0.5 * t_h

							if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
								 ( t_y <= y_bar   <= (t_y + t_h)) and 
								 ( left   <= t_x_bar <= (left   + w  )) and 
								 ( top   <= t_y_bar <= (top   + h  ))):
								matchedFid = fid

						if matchedFid is None:


							tracker = dlib.correlation_tracker()
							tracker.start_track(process_frame,
												dlib.rectangle( left,
																top,
																right,
																bottom))
							if face_ids==None:
								face_ids=0

							faceTrackers[ face_ids ] = tracker
							faceScores[ face_ids ] = confidence
							face_ids+=1

				# points = results[1]
				# print(results)
	 
		# display the size of the queue on the frame
		# cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
		# 	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
	 
		# show the frame and update the FPS counter
		for fid in faceTrackers.keys():
			tracked_position =  faceTrackers[fid].get_position()

			t_x = int(tracked_position.left())
			t_y = int(tracked_position.top())
			t_w = int(tracked_position.width())
			t_h = int(tracked_position.height())

			o_x   = ((   t_x   ) * frame.shape[1]/ process_frame.shape[1] );
			o_y    = ((   t_y    ) *  frame.shape[0]/process_frame.shape[0]);
			

			o_x2  = ((t_x+t_w  + 1) *  frame.shape[1]/process_frame.shape[1] ) - 1;
			o_y2 = ((t_y+t_h + 1) *  frame.shape[0]/process_frame.shape[0]) - 1;
			


			cv2.rectangle(frame, (int(o_x), int(o_y)),
									(int(o_x2), int(o_y2)),
									rectangleColor ,2)
		cv2.imshow("Frame", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		fps.update()
	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	 
	# do a bit of cleanup
	cv2.destroyAllWindows()
	fvs.stop()
