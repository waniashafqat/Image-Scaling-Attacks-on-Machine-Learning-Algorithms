# LO Norm
python ./attack.py --sourceImg 70.png --targetImg turn.jpg --outputImg outputl0.jpg --attackImg attackl0.jpg --resizeFunc cv2.resize --interpolation cv2.INTER_LINEAR --penalty 0.1 --imageFactor 255 --norm l0

# L2 Norm
python ./attack.py --sourceImg 70.png --targetImg turn.jpg --outputImg outputl2.jpg --attackImg attackl2.jpg --resizeFunc cv2.resize --interpolation cv2.INTER_LINEAR --penalty 0.1 --imageFactor 255 --norm l2

# LI Norm
#python ./attack.py --sourceImg 70.png --targetImg turn.jpg --outputImg outputli.jpg --attackImg attackli.jpg --resizeFunc cv2.resize --interpolation cv2.INTER_LINEAR --penalty 0.01 --imageFactor 255 --norm li
