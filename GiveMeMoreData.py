__author__ = ['Zumo Arthicha Srisuchinnawong']
__version__ = 2.0


''' import module '''
import os
from os import listdir
from os.path import isfile, join
import random
import copy

import cv2
import numpy as np

from module.IP_ADDR import Image_Processing_And_Do_something_to_make_Dataset_be_Ready as ipaddr

# set printing resolution of numpy module
np.set_printoptions(threshold=np.inf)


Main_Path = os.getcwd()

Font_Path = ["ENGFONT\\","ENGFONT\\","THFONT\\"]
Word = ["NUM","EN","TH"]




Save_Path = "D:\\2560\\FRA361_Robot_Studio\\FIBO_project_Module8-9\\Dataset\\Tew\\Augmented_dataset\\"
Font_Size = 32
AUGMOUNT = 50
Image_Shape = (64, 32)
MAGNIFY = [90,110]
MORPH = [1,5]
MOVE = [-3,3]
GAMMA = [10,40]



wordlist = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "0", "1", "2", "3", "4",
            "5", "6", "7", "8", "9", "ศูนย์ ", "หนึ่ง ", "สอง ", "สาม ", "สี่ ", "ห้า ", "หก ", "เจ็ด ", "แปด ",
            "เก้า "]
filename = {"zero": "zero", "one": "one", "two": "two", "three": "three", "four": "four", "five": "five",
            "six": "six", "seven": "seven", "eight": "eight", "nine": "nine", "0": "0", "1": "1", "2": "2",
            "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "ศูนย์ ": "ZeroTH",
            "หนึ่ง ": "OneTH", "สอง ": "TwoTH", "สาม ": "ThreeTH", "สี่ ": "FourTH", "ห้า ": "FiveTH",
            "หก ": "SixTH",
            "เจ็ด ": "SevenTH", "แปด ": "EightTH", "เก้า ": "NineTH"}

def RND_MAGNIFY(img,step = 5):
    global MAGNIFY
    magnify_value = np.arange(MAGNIFY[0],MAGNIFY[1],step).tolist()
    magnify_img = ipaddr.magnifly(img, percentage=magnify_value[random.randint(0,len(magnify_value)-1)])
    return magnify_img

def RND_MORPH(img,step = 1):
    global MORPH
    morph_value = np.arange(MORPH[0],MORPH[1],step).tolist()
    MODE = ipaddr.DILATE
    if random.randint(0,1):
        MODE = ipaddr.ERODE
    value = random.choice(morph_value)
    # morph_value[random.randint(0,len(morph_value)-1)],morph_value[random.randint(0,len(morph_value)-1)]
    morph_image =ipaddr.morph(img,MODE,value=[value,value])
    return morph_image

def RND_MOVE(img,step=2):
    global MOVE
    movex_value = np.arange(MOVE[0],MOVE[1],step).tolist()
    movey_value = np.arange(MOVE[0],MOVE[1],step).tolist()
    valuex = random.choice(movex_value)
    valuey = random.choice(movey_value)
    mov_image = ipaddr.translate(img, (valuex, valuey),[cv2.INTER_LINEAR, ipaddr.BORDER_CONSTANT, 255])
    return mov_image

def RND_GAMMA(image,step=1):
    global GAMMA
    value = np.arange(GAMMA[0],GAMMA[1],step).tolist()
    gamma = float(random.choice(value))/10.0
    if random.randint(0,1):
        gamma = 1.0/gamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def genny(font_path,font,wordlist):
    write = ''
    for x in wordlist:
        write = ""
        skip = False
        process = 0
        finish = len(font)
        for y in font:
            if process % (finish//4) == 0:
                #pass
                print('WORD:',x,'PROCESS:',process*100.0/finish, 'percent')
            process += 1
            # print(y)
            img = ipaddr.font_to_image(font_path + y, Font_Size, 0, x)

            #img = np.resize(img,(Image_Shape[1],Image_Shape[0]))
            plate = ipaddr.Get_Plate2(img)
            plate = ipaddr.Get_Word2(plate,image_size=Image_Shape)#get_plate(img, Image_Shape)
            img = np.array(plate[0])
            #ret, img = cv2.threshold(img, 200, 255,0)
            for i in range(0,AUGMOUNT):
                image = copy.deepcopy(img)
                #image = RND_MAGNIFY(image)
                image = RND_MORPH(image)
                #image = RND_MOVE(image)
                image = RND_GAMMA(image)
                stringy = np.array2string(((image.ravel())).astype(int), max_line_width=Image_Shape[0]*Image_Shape[1]*(AUGMOUNT+2),separator=',')
                write += stringy[1:-1] + "\n"
                if i == 0:
                    cv2.imshow('image',image)
                    cv2.waitKey(1)

            if process==len(font)*0.2:
                open(Save_Path+"dataset" + "_" + filename[x]+"_"+"test" + '.txt', 'w').close()
                file = open(Save_Path+"dataset"+"_"+filename[x] +"_"+"test"+ '.txt', 'a')
                file.write(write)
                file.close()
                write = ""
            elif process == len(font)*0.4:
                open(Save_Path+"dataset" + "_" + filename[x] + "_" + "validate" + '.txt', 'w').close()
                file = open(Save_Path+"dataset" + "_" + filename[x] + "_" + "validate" + '.txt', 'a')
                file.write(write)
                file.close()
                write = ""
            elif process == len(font)-1:
                open(Save_Path+"dataset" + "_" + filename[x] + "_" + "train" + '.txt', 'w').close()
                file = open(Save_Path+"dataset" + "_" + filename[x] + "_" + "train"+ '.txt', 'a')
                file.write(write)
                file.close()

for i in range(0,len(Font_Path)):


    font_path = Main_Path+Font_Path[i]
    word = Word[i]


    font = [x for x in listdir(font_path) if
                    ".ttf" in x or ".otf" in x or ".ttc" in x or ".TTF" in x or ".OTF" in x or ".TTC" in x]
    font =font[:-1]
    random.shuffle(font)

    if word is "EN":
        wl = wordlist[0:10]
    elif word is "NUM":
        wl = wordlist[10:20]
    elif word is "TH":
        wl = wordlist[20:]
    genny(font_path,font,wl)
