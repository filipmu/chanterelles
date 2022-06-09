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
print("Started")
from fastai.vision import *
from picamera import PiCamera  #for Pi
from time import sleep
from io import BytesIO
import sounddevice as sd

import PySimpleGUI as sg
import PIL
#import cv2 #for Desktop
from time import sleep, strftime

start_idx = 0
    
def main(args):

    path = Path("/home/filip/Mushrooms")  ##for rpi

    #path = Path('/media/SSD/Mushrooms')  ##for desktop
    path_train = path / 'train'
    camera=PiCamera()
    camera.resolution = (1024,768)

    print("camera initialized...")
    learn = load_learner(path_train,"squeezenet11_export.pkl");
    learn.to_fp32();
    print("model loaded...")


    ## uses a ram io file to store image to get to PySimpleGUI
    ## saves incorrect images

    def take_picture():
        ##for desktop
        #camera = cv2.VideoCapture(0) ##for desktop
        #ret = camera.set(3,224) ##for desktop
        #ret = camera.set(4,224) ##for desktop
        #s, image = camera.read() ##for desktop
        #camera.release() ##for desktop            
        #image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  ##for desktop
        #image = PIL.Image.fromarray(image)

        stream=BytesIO()  ##for rpi
        camera.capture(stream,format='png',resize=(224,224))  ##for rpi
        stream.seek(0)  ##for rpi
        image=PIL.Image.open(stream).convert('RGB')  ##for rpi
        imdata = stream.getvalue()
        return image,imdata

    image,imdata = take_picture()

    


    layout = [  [sg.Text('Mushroom Type', size=(20, 1), font=('Helvetica', 12), justification='center')],
            [sg.Image(data = imdata, key='_IMAGE_')],
            [sg.Text('', size=(20, 1), font=('Helvetica', 12), justification='center', key='_CLASS_')],
            [sg.Text('', size=(20, 1), font=('Helvetica', 12), justification='center', key='_PROB_')],
            [sg.Button('Correction', focus=True), sg.Quit()]]

    window = sg.Window('Mushroom AI', layout,no_titlebar=True, location=(0,0), keep_on_top=True).Finalize()


    window.Maximize()

    event, values = window.Read(timeout=10)

    amplitude = .5
    frequency = 1000
    duration = 0.1
    modulationf=1
   

    def sinewave(t,amplitude = 0.5,frequency = 0,phase = 0):
        return amplitude*np.sin(phase+2 * np.pi * frequency * t)

    def gate(t,duration):
        return (t<duration)

    def periodicgate(t,frequency,delay = 0.25, stop=0.50):
        x=((t* frequency) % 1)/frequency
        return (x>=delay) & (x<=stop)

    def sawtoothwave(t,frequency,delay=0):
        return (delay+  frequency * t) % 1 


    samplerate = 16000

    def callback(outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        global start_idx
        t = (start_idx + np.arange(frames)) / samplerate
        t = t.reshape(-1, 1)
        outdata[:] = sinewave(t,frequency=frequency) * periodicgate(t,modulationf,0,.05)
        start_idx += frames

    while True:
        with sd.OutputStream( channels=1, callback=callback,
                             samplerate=samplerate):

            event, values = window.Read(timeout=10) # Please try and use as high of a timeout value as you can
            if event is None or event == 'Quit':    # if user closed the window using X or clicked Quit button
                window.Close()
                break


            if event == 'Correction' and str(pred_class) == 'chanterelles':
                image.save(str(path/'corrections/other/not_chant') + strftime('%Y-%m-%d-%X')+str('.jpg'),'JPEG')

            if event == 'Correction' and str(pred_class) == 'other':
                image.save(str(path/'corrections/chanterelles/chant') + strftime('%Y-%m-%d-%X')+str('.jpg'),'JPEG')


            image,imdata = take_picture()
            img=Image(pil2tensor(image,np.float32).div_(255))

            pred_class,pred_idx,outputs = learn.predict(img)
            sleep(.5)
            prob=outputs[0].tolist()
            #print(pred_class,prob)

  


            window.Element('_IMAGE_').Update(data=imdata )
            window.Element('_PROB_').Update(f'prob: {(prob*100.0):02.2f} %')
            window.Element('_CLASS_').Update(str(pred_class))
    return 0




if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
