import numpy as np
import cv2 as cv
# Create a black image
#img = np.ones((512,512,3), np.uint8)*128 # gray background

#print(img)
# Draw a diagonal blue line with thickness of 5 px
#cv.line(img,(0,0),(511,511),(255,0,0),5)

# drawing a rectangle:
#cv.rectangle(img,(384,0),(510,128),(0,255,0),3)

# drawing a circle:
#cv.circle(img,(447,63), 63, (0,0,255), -1)

import music21 as m21
from itertools import product

pitchToZ12 = dict(zip(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"],range(12)))
Z12ToPitch = dict(zip(range(12),["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]))

import numpy as np


def xml_to_list(xml):
    xml_data = m21.converter.parse(xml)
    score = []
    for part in xml_data.parts:
        parts = []
        for note in part.recurse().notesAndRests:
            if note.isRest:
                start = note.offset
                duration = float(note.quarterLength)/4.0
                vol = 32 #note.volume.velocity
                pitches= tuple([-1])
                parts.append(tuple([pitches,duration,vol,1]))
            elif type(note)==m21.chord.Chord:
                pitches = sorted([e.pitch.midi for e in note]) # todo: think about chords
                vol = int(note[0].volume.velocity)
                duration = float(note.duration.quarterLength)/4.0
                parts.append(tuple([tuple(pitches),duration,vol,0]))
            else:
                #print(note)
                start = note.offset
                duration = float(note.quarterLength)/4.0
                pitches = tuple([note.pitch.midi])
                #print(pitch,duration,note.volume)
                vol = note.volume.velocity
                if vol is None:
                    vol = int(note.volume.realized * 127)
                parts.append(tuple([pitches,duration,vol,0]) )
        score.append(parts)        
    return score


def parseXml(fp):
    return xml_to_list(fp)


def draw_circle(image, pp, color, radius=0):
    x,y = pp
    image = cv.circle(image, (x,y),  color=color, thickness=cv.FILLED,radius=radius)
    return image

def draw_line(image, start,end, color):
    image = cv.line(image, start, end, color, thickness=2)
    return image

#img = draw_point(img,(256,256),(255,255,255))

def ff(a=1,b=6,c=-14,x=1,y=1/2,z=np.complex(0,1)/3):
    i = np.complex(0,1)
    return (lambda t: x*np.exp(a*i*t)+y*np.exp(b*i*t)+z*np.exp(c*i*t))

def FF(nn = [1,3],m=2, k=1, aa=[1,2]):
    if all([n % m == k for n in nn]) and len(aa)==len(nn) and np.gcd(k,m)==1:
        i = np.complex(0,1)
        return (lambda t: sum([ aa[j]*np.exp(nn[j]*i*t) for j in range(len(aa)) ]))
    else:
        return None

def draw_curve(img, ff, mm, color, rr=120,number_of_points = 100):

    def compute_point(ff,k,start,step,rr,mm):
        t = start+k*step
        z = ff(t)
        x,y = z.real,z.imag
        # scale:
        x = x*rr
        y = y*rr
        # translate:
        x,y = x+mm[0],y+mm[1]
        # round to integers
        x,y = int(x),int(y)
        return x,y    
        
    start = 0.0
    end = 2*np.pi*14
    N = number_of_points
    step = (end-start)/N
    for k in range(N-1):
        x,y = compute_point(ff,k,start,step,rr,mm)
        x2,y2 = compute_point(ff,k+1,start,step,rr,mm)
        #print(x,y)
        #
        img = draw_line(img, (x,y),(x2,y2), color=color)
    return img


def color_img(img):
    ret, thresh = cv.threshold(img, 127, 255, 0)
    num_labels, labels = cv.connectedComponents(thresh,connectivity=8)

    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    return labeled_img

#img = draw_circle(img,(256,256), 63, (0,0,255),number_of_points = 10000)


def getImgNrs(start_duration,end_duration,bpm,fps):
    N_img_start = int(np.floor(4*fps*60*start_duration/bpm))
    N_img_end = int(np.ceil(4*fps*60*end_duration/bpm))
    return (N_img_start,N_img_end)
    
def convertScore(scores,bpm=70,fps=25,verbose=False):
    #determine max durations:
    maxDurs = []
    pitchSet = set([])
    volumeSet = set([])
    for part in scores:
        maxDurs.append(0)
        for note in part:
            pitches, duration, volume, rest = note
            maxDurs[-1] += duration
    maxDur = int(np.ceil(max(maxDurs)))
    print(maxDur)
    Nimgs = int(np.floor(4*60*fps*maxDur/bpm))
    imgs2Notes = dict([])
    #fill dictionary with notes per image
    for part in scores:
        dur = 0
        for note in part:
            pitches, duration, volume, rest = note
            pitchSet.add(pitches[0])
            volumeSet.add(volume)
            start_img, end_img = getImgNrs(start_duration=dur,end_duration = dur+duration,bpm=bpm,fps=fps)
            if verbose: print(note,start_img,end_img)
            for k in range(start_img,end_img+1):
                if k in imgs2Notes.keys():
                    imgs2Notes[k].append(note)
                else:
                    imgs2Notes[k] = [note]
            dur += duration
    return imgs2Notes,pitchSet,volumeSet      

def create_video(imgs,videoname="./opencv_videos/video.avi",fps=25):
    fourcc = cv.VideoWriter_fourcc(*"X264") 
    height,width,x = imgs[0].shape
    print(width,height,x,fps,videoname)
    framesPerSecond = fps
    video = cv.VideoWriter(videoname, fourcc, framesPerSecond, (width, height))
    cnt = 0
    for img in imgs:
        #print(cnt,img.shape)
        video.write(img)
        cnt += 1
    video.release()
    return video

def compute_color(pitch,volume,t,N,noteCounter,lN):
    return (int(pitch*2*np.sin(2*np.pi*t/N)),int(volume*2*np.sin(2*np.pi*t/N)),int((noteCounter/lN)*128))
    
def make_video(imgs2Notes,pitchSet,volumeSet,videoname="./opencv_videos/video.avi",fps=25,verbose=False):
    fourcc = cv.VideoWriter_fourcc(*"X264") 
    height,width = 512,512
    print(width,height,fps,videoname)
    framesPerSecond = fps
    video = cv.VideoWriter(videoname, fourcc, framesPerSecond, (width, height))
    cnt = 0

 
    N = len(imgs2Notes.keys())
    import random
    mv = min(volumeSet)
    Mv = max(volumeSet)
    dv = Mv-mv
    mp = min(pitchSet)
    Mp = max(pitchSet)        
    dp = Mp-mp
    dx = 50
    dy = 50
    breite = 512-2*dx
    hoehe = 512-2*dy
    for t in range(N):
        img = np.ones((512,512,3), np.uint8)*255 # white background
        notes = imgs2Notes[t]
        noteCounter = 0
        lN = len(notes)
        if verbose: print(t,"/",N," img")
        for note in notes:
            pitches, duration, volume,rest = note
            volumeScaled = (volume-mv)/dv
            pitchScaled = (pitches[0]-mp)/dp
            #print(t,note)
            pitch = pitches[0]
            x,y = dx+int(pitchScaled*breite),dy+int(volumeScaled*hoehe)
            #ff = FF(nn=[(volume//64)*m+k,(pitch//64)*m+k,2],m=m,k=k,aa=[(volume+pitch)/x*np.sin(t*np.pi*2/N)/2.0 for x in [128,128,128]])
            #img = draw_curve(img,ff,(x,y),(0,0,0),rr=10,number_of_points = 1000) # 011.png
            
            img = draw_circle(img,pp=(x,y), color = compute_color(pitch,volume,t,N,noteCounter,lN), radius=int((volume+pitch)//8))
            noteCounter += 1
        #labeled_img = color_img(img)
        #imgs.append(labeled_img)
        video.write(img)
    video.release()    
    return True

import sys

print(sys.argv)

midi = sys.argv[1] #"./mix_of_midis/midi/markov-3_120bpm_1min_muttertag.mid"
scores = parseXml(fp=midi)            
fps = int(sys.argv[2])
bpm = int(sys.argv[3])
print("fps = ", fps, "bpm = ", bpm)
imgs2Notes,pitchSet,volumeSet = convertScore(scores,bpm=bpm,fps=fps)
#print(volumeSet)
#print(imgs2Notes)
vn = sys.argv[4]
make_video(imgs2Notes,pitchSet,volumeSet,videoname=vn,fps=fps,verbose=True)    

