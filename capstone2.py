from calendar import c
from msilib.schema import Class
from pydoc import classname
import torch
import torchvision
from glob import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from enum import Enum

class ClassNum(Enum):
    PERSON = 0
    BAG1 = 7
    BAG2 = 57
    BAG3 = 24
    BAG4 = 28
    BAG5 = 26
    TABLE1 = 60
    TABLE2 = 13
    

#def visitStat(result):

def visitTime(stat):
    tm = time.localtime(time.time())
    filename = time.strftime('%Y-%m-%d.txt',tm)
    
    if tm.min % 3 == 0:
        with open(filename,"a") as f:
            f.write(time.strftime("%Y-%m-%d-%H-%M\n",tm))
            #f.write(stat.)
"""
n분마다 파일 쓰기로 설정
현시각 여자 청년, 중년, 노년 남자 청년 중년 노년 분류

"""

def occupiedCheck(tableList, personList, itemList):
    tableCheck = []
    print(len(tableList))
    cnt = 0
    for table in tableList:
        tableCheck.append(0)
        for person in personList:
            if table[0]<=((person[0]+person[1])/2)<=table[2]:
                if table[1]<=((person[1]+person[3])/2)<=table[3]:
                    tableCheck[cnt] = 2
        for item in itemList:
            if table[0]<=((item[0]+item[1])/2)<=table[2]:
                if table[1]<=((item[1]+item[3])/2)<=table[3]:
                    if tableCheck[cnt] == 0:
                        tableCheck[cnt] = 1
        cnt = cnt+1

        
            
            
    return tableCheck, int(time.time())
"""
1. 테이블을 y축 기준으로 정렬
2. 테이블 좌표 안에 사람이 존재할 경우 -> 테이블 설정
3. 테이블 좌표 안에 가방만 존재할 경우 -> 테이블 자리비움 설정
4. 테이블 좌표 안에 아무것도 없을 경우 -> 빈테이블 설정
"""

def positionCheck(boxes,img):
    personList = [] 
    tableList = []
    itemList=[]
    for person in boxes.values:
        if person[5] == ClassNum.PERSON.value:
            bais = -40
            person[0] = int(person[0]) - bais
            person[1] = int(person[1]) - bais
            person[2] = int(person[2]) + bais
            person[3] = int(person[3]) + bais
            personList.append(person)
            cv.rectangle(img, (person[0],person[1]),(person[2],person[3]),(0,0,255),1) #cv로 객체 위치 그리기
    for table in boxes.values:
        if table[5] == ClassNum.TABLE1.value:
            bais = 50
            table[0] = int(table[0]) - bais
            table[1] = int(table[1]) - bais
            table[2] = int(table[2]) + bais
            table[3] = int(table[3]) + bais
            tableList.append(table)
            cv.rectangle(img, (table[0],table[1]),(table[2],table[3]),(255,1,15),2)
    
    for item in boxes.values:
        if item[5] == ClassNum.BAG1.value or item[5] == ClassNum.BAG2.value or item[5] == ClassNum.BAG3.value or item[5] == ClassNum.BAG4.value or item[5] == ClassNum.BAG5.value:
            bais = -40
            item[0] = int(item[0]) - bais
            item[1] = int(item[1]) - bais
            item[2] = int(item[2]) + bais
            item[3] = int(item[3]) + bais
            itemList.append(item)
            cv.rectangle(img, (item[0],item[1]),(item[2],item[3]),(125,100,0),2)
    return personList, tableList, itemList


"""def positionCheck(boxes,img,cn = None):
    objList = [] 
    if cn == ClassNum.TABLE.value:
        bais = 50
    elif cn == ClassNum.PERSON.value:
        bais = -40
    elif cn == None:
        bais = 0
    else:
        bais = -40
    for obj in boxes.values:
        if obj[5] == cn:
            obj[0] = int(obj[0]) - bais
            obj[1] = int(obj[1]) - bais
            obj[2] = int(obj[2]) + bais
            obj[3] = int(obj[3]) + bais
            objList.append(obj)
            cv.rectangle(img, (obj[0],obj[1]),(obj[2],obj[3]),(0,255,0),1) #cv로 객체 위치 그리기
    return objList
 """   
def writetext(tableCheck):

    with open("table.txt") as f:
        f.write(str(tableCheck))


def jpgCheck():
    print("table value : ",ClassNum.TABLE1.value, ClassNum.TABLE2.value)

    _path = 'C:/Users/Yun/Document/python/yolov5/yolov5l.pt'
    img_path = 'C:/Users/Yun/Documents/python/yolov5/img.jpg'
    
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=_path)  # local model
    img = cv.imread(img_path)
    results = model(img_path)

    boxes = results.pandas().xyxy[0]
    boxes = boxes.sort_values('ymin')
    print("==boxes==\n",boxes)
    
    personList, tableList, itemList = positionCheck(boxes,img)
    cv.imshow('camera', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("tableList :",tableList)
    print("personList :",personList)
    print("itemList : ",itemList)

    tableCheck, tm = occupiedCheck(tableList, personList, itemList)
    print("테이블 체크 :", tableCheck)
    print("체크한 시간 : ",tm)

def realtime():
    cap = cv.VideoCapture(0)
    _path = 'C:/Users/Yun/Document/python/yolov5/yolov5l.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=_path)  # local model
    timeCheck = time.time()
    tableCheck = []

    while(True):
        ret, cam = cap.read()
        
        if(ret):
            result = model(cam)
            boxes = result.pandas().xyxy[0]
            boxes = boxes.sort_values('ymin')

            
            cv.imshow('camera', cam)

            personList, tableList, itemList = positionCheck(boxes,cam)
            print("tableList :",tableList)
            print("personList :",personList)

            tableCheck, tm = occupiedCheck(tableList, personList, itemList)
            print("테이블 체크 :", tableCheck)
            print("체크한 시간 : ",tm)
            # writeText(tableCheck)
            if cv.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
                break

prevTime = 0
#realtime()
jpgCheck()




#토치 버전 다운 필요

        

"""
        curTime = time.time()
        if (curTime - timeCheck)>10:
            timeCheck = curTime

        tm = time.gmtime()
        if tm.tm_min % 3 == 0 and check:
            print(boxes)
            check = False
        if tm.tm_min % 4 == 0 :
            check == True
    
"""        
    # 날짜 측정 -> 날짜 형태 리스트로 -> 시간대별로 
    #time.strftime('%c',time.localtime(time.time()))

