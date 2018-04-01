__author__ = ['Zumo Arthicha Srisuchinnawong']
__version__ = 2.0
__description__ = 'Data Generaing Program'

'''*************************************************
*                                                  *
*                 import module                    *
*                                                  *
*************************************************'''

import os
from os import listdir
import copy
from module.RandomFunction import *

# set printing resolution of numpy module
np.set_printoptions(threshold=np.inf)


'''*************************************************
*                                                  *
*                 global variable                  *
*                                                  *
*************************************************'''
MAIN_PATH = os.getcwd()
Font_Path = ["\\font\\ENGFONT\\","\\font\\ENGFONT\\","\\font\\THFONT\\"]
Word = ["NUM","EN","TH"]


SAVE_PATH = MAIN_PATH + "\\dataset\\synthesis\\textfile\\"
Font_Size = 32
AUGMOUNT = 30
Image_Shape = (64, 32)
MAGNIFY = [90,110]
MORPH = [1,5]
MOVE = [-3,3]
GAMMA = [10,40]

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

FILENAME = range(0,30)
FILENAME = list(map(str, FILENAME))
WORDLIST = ["0","1","2","3","4","5","6","7","8","9",
            "zero","one","two","three","four","five","six","seven","eight","nine",
            "ศูนย์","หนึ่ง","สอง","สาม","สี่","ห้า","หก","เจ็ด","เเปด","เก้า"]

def Gennie(font_path,font,wordlist,waitTime=1,start=0):

    '''
    :param font_path: path to font
    :param font: font name
    :param wordlist: word
    :param waitTime: wait time each frame to show
    :param start: index of the first word in each class (0 for number, 10 for english word and
    20 for this word)
    :return: None
    example
            Gennie('c:\font\ENGFONT','bell.ttf',['one','two'],start=0)
        this function will search for font 'bell' inside directory 'c:\font\ENGFONT' (ENGFONT start at 0), then save two compressed
    text file of 'one' and 'two' in directory specified by global variable SAVE_PATH.
    '''

    global FILENAME,SAVE_PATH

    for w in range(0,len(wordlist)):
        word = wordlist[w]
        write = ""
        process = 0
        finish = len(font)
        for y in font:
            if process % (finish//4) == 0:
                print('WORD:',word,'PROCESS:',process*100.0/finish, 'percent')
            process += 1
            img = ipaddr.font_to_image(font_path + y, Font_Size, 0, word)
            plate = ipaddr.Get_Plate2(img)
            plate = ipaddr.Get_Word2(plate,image_size=Image_Shape)
            img = np.array(plate[0])

            for i in range(0,AUGMOUNT):
                image = copy.deepcopy(img)
                #image = RND_MAGNIFY(image,MAGNIFY)
                image = RND_MORPH(image,MORPH)
                #image = RND_MOVE(image,MOVE)
                image = RND_GAMMA(image,GAMMA)
                stringy = np.array2string(((image.ravel())).astype(int), max_line_width=Image_Shape[0]*Image_Shape[1]*(AUGMOUNT+2),separator=',')
                write += stringy[1:-1] + "\n"
                if i == 0:
                    cv2.imshow('image',image)
                    cv2.waitKey(waitTime)
            if process==len(font)*0.2:
                open(SAVE_PATH+FILENAME[w+start]+"_"+"test" + '.txt', 'w').close()
                file = open(SAVE_PATH+FILENAME[w+start] +"_"+"test"+ '.txt', 'a')
                file.write(write)
                file.close()
                write = ""
            elif process == len(font)*0.4:
                open(SAVE_PATH+FILENAME[w+start] + "_" + "validate" + '.txt', 'w').close()
                file = open(SAVE_PATH+ FILENAME[w+start] + "_" + "validate" + '.txt', 'a')
                file.write(write)
                file.close()
                write = ""
            elif process == len(font)-1:
                open(SAVE_PATH+FILENAME[w+start] + "_" + "train" + '.txt', 'w').close()
                file = open(SAVE_PATH+FILENAME[w+start] + "_" + "train"+ '.txt', 'a')
                file.write(write)
                file.close()

for i in range(0,len(Font_Path)):
    # loop through NUM, ENG and THAI
    font_path = MAIN_PATH+Font_Path[i]
    word = Word[i]
    font = [x for x in listdir(font_path) if
                    ".ttf" in x or ".otf" in x or ".ttc" in x or ".TTF" in x or ".OTF" in x or ".TTC" in x]
    font =font[:-1]
    random.shuffle(font)

    if word is "EN":
        wordlist = WORDLIST[10:20]
        start = 10
    elif word is "NUM":
        wordlist = WORDLIST[0:10]
        start = 0
    elif word is "TH":
        wordlist = WORDLIST[20:]
        start = 20


    Gennie(font_path,font,wordlist,start=start)
