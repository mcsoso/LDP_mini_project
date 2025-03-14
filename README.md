# LDP_mini_project

python3 -m venv .env  
source .env/bin/activate  
pip install -r requirements.txt  

create a folder "annotated" save the image with annotated pumpkins in it (annotations in bgr[43, 21, 229]).  
And also the same image without annotations.  

create a folder "pictures" and save there the images on which you want to count the pumpkins  

run annotated.py 
results are stored in the folder "pumpkin_results"  
  
run count.py
results are stored in "pumpkin_detection_results"
