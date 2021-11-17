#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:51:03 2021

pip install SpeechRecognition

@author: mwmak
"""

import speech_recognition as sr
import glob
import os

def recognize(sphfile):
    r = sr.Recognizer()
    audiofile = sr.AudioFile(sphfile)
    with audiofile as source:
        speech = r.record(source)
        return(r.recognize_google(speech))

def get_target1(tranfile, key):
    with open(tranfile) as f:
        for line in f:
            fields = line.split(sep=' ', maxsplit=1)
            if key == fields[0]:
                return fields[-1].rstrip()

def get_target2(tranfile):
    with open(tranfile) as f:
        return(f.readline().split(sep=' ', maxsplit=2)[-1].rstrip())
        

if __name__ == '__main__':
    filelist = glob.glob('data/**/*.wav', recursive=False)
    for file in filelist:
        rec_text = recognize(file)
        tranfile = file.replace('.wav', '.txt').replace('/speech/','/text/')
        tgt_text = get_target2(tranfile)
        print(f"Target: {tgt_text}")
        print(f"Result: {rec_text}\n")

    filelist = glob.glob('data/**/*.flac', recursive=False)
    for file in filelist:
        base = os.path.basename(file)
        key = base.replace('.flac', '')
        rec_text = recognize(file)
        tgt_text = get_target1("data/text/1272-128104.trans.txt", key)
        print(f"Target: {tgt_text}")
        print(f"Result: {rec_text}\n")
    
    
    
    
    
