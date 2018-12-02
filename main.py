import cv2
import sys
from openalpr import Alpr
from filter import filter_mask
from argparse import ArgumentParser
import os
import csv
import datetime



parser = ArgumentParser(description='OpenALPR Python Program')

parser.add_argument("-c", "--country", dest="country", action="store", default="us",
                  help="License plate Country" )

parser.add_argument("--config", dest="config", action="store", default="C:/openALPR_1/openalpr_64/openalpr.conf",
                  help="Path to openalpr.conf config file" )

parser.add_argument("--runtime_data", dest="runtime_data", action="store", default="C:/openALPR_1/openalpr_64/runtime_data",
                  help="Path to OpenALPR runtime_data directory" )

parser.add_argument('Video_Source', help='License plate image file')

parser.add_argument('thrsArea', help='Threshold pixel area for moving vehicle detcetion')

options = parser.parse_args()

dst=os.getcwd()+'\cropedImages'         # destination to save the images
if not os.path.exists(dst):
    os.makedirs(dst)

fieldnames = ['ID', 'Plate number', 'Image Path','Date' , 'Time']
fileID=open('file.csv','w')       # file to save the data
newFileWriter = csv.writer(fileID)
newFileWriter.writerow(fieldnames)

alpr = Alpr(options.country, options.config, options.runtime_data)

#alpr = Alpr("mx", "C:/openALPR_1/openalpr_64/openalpr.conf", "C:/openALPR_1/openalpr_64/runtime_data")
#alpr.set_top_n(10)
alpr.set_default_region("md")
if not alpr.is_loaded():
    sys.exit(1)

print("Using OpenALPR " + alpr.get_version())
cap = cv2.VideoCapture(options.Video_Source)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, detectShadows=True)

print ('Training BG Subtractor...')
cv2.namedWindow('op', cv2.WINDOW_NORMAL)
cnt=0
while True:
    ok,frame=cap.read()
    if not ok:

        sys.exit()
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        Rfilter = cv2.bilateralFilter(gray, 9, 75, 75)

        # Threshold image
        ret, filtered = cv2.threshold(Rfilter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        filtered = cv2.medianBlur(frame, 5)
        fg_mask = bg_subtractor.apply(filtered, None, 0.01)
        fg_mask = filter_mask(fg_mask)
        im, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cntr = []
        for contour in contours:
            contourSize = cv2.contourArea(contour)
            if (contourSize > int(options.thrsArea)):
                cntr.append(contour)
                cnt+=1

        for ii in range(0, len(cntr)):
            (x, y, w, h) = cv2.boundingRect(cntr[ii])

            cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1),
                          (0, 255, 0), 1)
            cropedframe=frame[y:y+h, x:x+w]
            filename=dst+'\\'+str(cnt)+'.jpg'
            cv2.imwrite(filename,cropedframe) # write the image

            ret, enc = cv2.imencode("*.bmp", cropedframe)
            results = alpr.recognize_array(bytes(bytearray(enc)))

            if results['results']:
                print('License plate Detected and Recorded')

                time = datetime.datetime.now().time()
                date = datetime.datetime.now().date()
                plateno = results['results'][0]['plate']
                confidance=results['results'][0]['confidence']
                print('Plate: ',plateno, 'Confidance: ',confidance)

                newFileWriter.writerow([cnt, str(plateno), filename, date, time]) # write the csv file


        cv2.imshow('op', frame)
        if cv2.waitKey(33) == 27:
            break
