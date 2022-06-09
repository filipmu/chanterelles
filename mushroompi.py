#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  mushroompi.py
#  
#  Copyright 2019 Filip Mulier <filip@filip-pi>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

from fastai.vision import *
from picamera import PiCamera
from time import sleep
from io import BytesIO
import sounddevice as sd



def main(args):
	print("Started")
	path = Path("/home/filip/Mushrooms")
	path_train = path / 'train'
	camera=PiCamera()
	camera.resolution = (1024,768)
	#camera.resolution = (320,240)
	#camera.start_preview()
	print("camera initialized...")
	learn = load_learner(path_train,"squeezenet11_export.pkl")
	learn.to_fp32();
	print("model loaded...")
	
	fs=44100
	sound_sample=(np.sin(2*np.pi*np.arange(fs*0.25)*2000/fs)).astype(np.float32)
	
	while True:
		#camera.capture('/home/filip/Mushrooms/temp.jpg',resize=(224,224))
		#img = open_image('/home/filip/Mushrooms/temp.jpg')
		stream=BytesIO()
		camera.capture(stream,format='jpeg',resize=(224,224))
		stream.seek(0)
		image=PIL.Image.open(stream).convert('RGB')
		
		img=Image(pil2tensor(image,np.float32).div_(255))
		#print("image captured and saved...")
		
		#image=np.empty((240,320,3),dtype=np.uint8)
		#camera.capture(image, 'rgb')
		#img=Image(tensor(image))
		pred_class,pred_idx,outputs = learn.predict(img)
		prob=outputs[0].tolist()
		print(pred_class,prob)
		sd.play(sound_sample*prob,fs)
	
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
