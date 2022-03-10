import numpy as np
import cv2 as cv
# Create a black image
#img = np.ones((512,512,3), np.uint8)*128 # gray background
from sage.all import *
import numpy as np

import pandas,sys

import statsmodels.api as sm
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
        print(part)
        for note in part.flat.notesAndRests:
            if type(note)==m21.note.Rest:
                print("rest", note, note.duration.quarterLength)
                duration = float(note.duration.quarterLength)
                vol = 32 #note.volume.velocity
                pitches= tuple([64])
                parts.append(tuple([float(note.offset),pitches,duration,vol,1]))
            elif type(note)==m21.chord.Chord:
                print("chord ",note,note.duration.quarterLength)
                pitches = sorted([e.pitch.midi for e in note]) # todo: think about chords
                vol = note[0].volume.velocity
                if vol is None:
                    vol = int(note[0].volume.realized * 127)
                else:
                    vol = int(vol)    
                duration = float(note.duration.quarterLength)
                parts.append(tuple([float(note.offset),tuple(pitches),duration,vol,0]))
            else:
                print("note", note,note.duration.quarterLength)
                start = note.offset
                duration = float(note.quarterLength)
                pitches = tuple([note.pitch.midi])
                #print(pitch,duration,note.volume)
                vol = note.volume.velocity
                if vol is None:
                    vol = int(note.volume.realized * 127)
                parts.append(tuple([float(note.offset),pitches,duration,vol,0]) )
        score.append(parts) 
    print( [ len(part) for part in score])           
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

def draw_curve(img, ff, mm, color, rr=120,number_of_points = 100,return_points = False):

    points = []
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
        points.append((x,y))
        img = draw_line(img, (x,y),(x2,y2), color=color)
    points.append((x2,y2))    
    if return_points: return img,points
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
    N_img_start = int(np.round(fps*60*start_duration/(bpm),0))
    N_img_end = int(np.round(fps*60*end_duration/bpm,0))
    return (N_img_start,N_img_end)
    
def convertScore(scores,bpm=70,fps=25,verbose=False):
    #determine max durations:
    maxDurs = [0 for k in range(len(scores))]
    startsAndDurs = [0 for k in range(len(scores))]
    pitchSet = set([])
    volumeSet = set([])
    partCounter = 0 
    for part in scores:
        for note in part:
            start,pitches, duration, volume, rest = note
            maxDurs[partCounter] += duration
            if startsAndDurs[partCounter] < start+duration:
                startsAndDurs[partCounter] = start+duration 
        partCounter+=1    
            
    maxDur = np.max(startsAndDurs)
    print(startsAndDurs)
    print(maxDur)
    print(bpm)
    Nimgs = int(np.round(60*fps*maxDur/bpm,0))
    print(Nimgs)
    imgs2Notes = dict([])
    #fill dictionary with notes per image
    for part in scores:
        dur = 0
        for note in part:
            start,pitches, duration, volume, rest = note
            print(start,pitches,duration,volume,rest)
            for pitch in pitches:
                pitchSet.add(pitch)
            volumeSet.add(volume)
            start_img, end_img = getImgNrs(start_duration=start,end_duration = start+duration,bpm=bpm,fps=fps)
            if verbose: print(note,start_img,end_img)
            for k in range(start_img,end_img+1):
                if k in imgs2Notes.keys():
                    imgs2Notes[k].append((note,start_img,end_img))
                else:
                    imgs2Notes[k] = [(note,start_img,end_img)]
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

def compute_color(pitch,volume,t,N,noteCounter,lN,start_img,end_img):
    tScaled = (t-start_img)/(end_img-start_img+1)
    return (int(tScaled*pitch*2*np.sin(2*np.pi*t/N)),int(tScaled*volume*2*np.sin(2*np.pi*t/N)),int(tScaled*(noteCounter/lN)*128))

def compute_radius(pitch,volume,t,N,noteCounter,lN,start_img,end_img):
    tScaled = (t-start_img)/(end_img-start_img+1)
    return max(1,int(tScaled*np.abs((volume)*np.cos(2*np.pi*t/N))))

#!/usr/bin/python

import numpy as np
import random

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw a point
def draw_point(img, p, color ) :
    cv.circle( img, p, 2, color, cv.FILLED, cv2.CV_AA, 0 )

# Draw voronoi diagram
def draw_voronoi(img, subdiv,color) :

    ( facets, centers) = subdiv.getVoronoiFacetList([])

    r,g,b = color

    
    lf = len(facets)

    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (255-i/lf*r, i/lf*g, i/lf*b)

        cv.fillConvexPoly(img, ifacet, color, cv.LINE_AA, 0);
        ifacets = np.array([ifacet])
        cv.polylines(img, ifacets, True, (0, 0, 0), 1, cv.LINE_AA, 0)
        cv.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv.FILLED, cv.LINE_AA, 0)
    return img    

def make_video_with_circles(imgs2Notes,pitchSet,volumeSet,videoname="./opencv_videos/video.avi",fps=25,verbose=False):
    fourcc = cv.VideoWriter_fourcc(*"X264") 
    height,width = 512,512
    print(width,height,fps,videoname)
    framesPerSecond = fps
    video = cv.VideoWriter(videoname, fourcc, framesPerSecond, (width, height))
    cnt = 0

    print("volumeSet = ",volumeSet) 
    print("pitchSet = ", pitchSet)
    N = len(imgs2Notes.keys())
    import random
    mv = min(volumeSet)
    Mv = max(volumeSet)
    dv = Mv-mv
    if dv ==0:
        dv = 1
    mp = min(pitchSet)
    Mp = max(pitchSet)        
    dp = Mp-mp
    if dp ==0:
        dp = 1        
    dx = 10
    dy = 10
    breite = 512-2*dx
    hoehe = 512-2*dy
    img = np.ones((512,512,3), np.uint8)*0 # white background
    rb0 = np.random.randint(1,breite)
    rh0 = np.random.randint(1,hoehe)
    
    pitchList = sorted(list(pitchSet))
    
    X0,invPitchDict = getCoordinatesOfPitchList(pitchList,dx,breite)
    
    
    for t in range(N):
        img = np.ones((512,512,3), np.uint8)*0 # white background
        rb = np.random.randint(-5*dx,5*dx)
        rh = np.random.randint(-5*dy,5*dy)
        notes = imgs2Notes[t]
        noteCounter = 0
        lN = len(notes)
        if verbose: print(t,"/",N," img")
        for tt in notes:
            note,start_img,end_img = tt
            start,pitches, duration, volume,rest = note
            if rest==1:
                continue
            #print(t,note)
            for pitch in pitches:
                volumeScaled = (volume-mv)/dv
                pitchScaled = (pitch-mp)/dp
                x,y = dx+int(pitchScaled*rb),dy+int(volumeScaled*rh)
                x0,y0 = [int(a) for a in X0[invPitchDict[pitch]]]
                x = x0 + rb
                y = y0 + rh
                radius = compute_radius(pitch,volume,t,N,noteCounter,lN,start_img,end_img)
                color = compute_color(pitch,volume,t,N,noteCounter,lN,start_img,end_img)
                img = draw_circle(img,pp=(x,y), color = color, radius=radius)
                noteCounter += 1
        noteCounter = 0  
        # zeichne helle kreise  
        for tt in notes:
            note,start_img,end_img = tt
            start,pitches, duration, volume,rest = note           
            #print(t,note)
            if rest==1:
                continue
            for pitch in pitches:
                pitchScaled = (pitch-mp)/dp
                volumeScaled = (volume-mv)/dv
                #ff = FF(nn=[(volume//64)*m+k,(pitch//64)*m+k,2],m=m,k=k,aa=[(volume+pitch)/x*np.sin(t*np.pi*2/N)/2.0 for x in [128,128,128]])
                #img = draw_curve(img,ff,(x,y),(0,0,0),rr=10,number_of_points = 1000) # 011.png
                radius = compute_radius(pitch,volume,t,N,noteCounter,lN,start_img,end_img)
                color = compute_color(pitch,volume,t,N,noteCounter,lN,start_img,end_img)
                x0,y0 = [int(a) for a in X0[invPitchDict[pitch]]]#dx+int(pitchScaled*breite),dy+int(volumeScaled*hoehe)
                m,k=3,2
                ff = FF(nn=[(volume//64)*m+k,(pitch//64)*m+k,2],m=m,k=k,aa=[(volume+pitch)/x*np.sin(t*np.pi*2/N)/2.0 for x in [128,128,128]])
                img = draw_curve(img,ff,(x0,y0),color=(255-color[0],255-color[1],255-color[2]),rr=radius,number_of_points = 100)
                #img = draw_circle(img,pp=(x0,y0),color=(255-color[0],255-color[1],255-color[2]),radius=radius)    
                noteCounter += 1
        #labeled_img = color_img(img)
        #imgs.append(labeled_img)
        video.write(img)
    video.release()    
    return True    

def make_video_with_voronoi(imgs2Notes,pitchSet,volumeSet,videoname="./opencv_videos/video.avi",fps=25,verbose=False):
    fourcc = cv.VideoWriter_fourcc(*"X264") 
    height,width = 512,512
    print(width,height,fps,videoname)
    framesPerSecond = fps
    video = cv.VideoWriter(videoname, fourcc, framesPerSecond, (width, height))
    cnt = 0

    print("volumeSet = ",volumeSet) 
    print("pitchSet = ", pitchSet)
    N = len(imgs2Notes.keys())
    import random
    mv = min(volumeSet)
    Mv = max(volumeSet)
    dv = Mv-mv
    if dv ==0:
        dv = 1
    mp = min(pitchSet)
    Mp = max(pitchSet)        
    dp = Mp-mp
    if dp ==0:
        dp = 1        
    dx = 10
    dy = 10
    breite = 512-2*dx
    hoehe = 512-2*dy
    img = np.ones((512,512,3), np.uint8)*255 # white background
    rb0 = np.random.randint(breite//2,breite)
    rh0 = np.random.randint(hoehe//2,hoehe)
    size = img.shape
    rect = (0,0,size[1],size[0])
    
    
    for t in range(N):
        #img = np.ones((512,512,3), np.uint8)*255 # white background
        rb = np.random.randint(1,breite)
        rh = np.random.randint(1,hoehe)
        try:
            notes = imgs2Notes[t]
        except:
            continue    
        noteCounter = 0
        lN = len(notes)
        if verbose: print(t,"/",N," img")
        subdiv = cv.Subdiv2D(rect);  
        # draw voronoi
        noteCounter = 0
        for tt in notes:
            note,start_img,end_img = tt
            start,pitches, duration, volume,rest = note
            volumeScaled = (volume-mv)/dv
            if rest==1:
                continue
            #print(t,note)
            for pitch  in pitches:
                pitchScaled = (pitch-mp)/dp
                #print(pitchScaled)
                radius = compute_radius(pitch,volume,t,N,noteCounter,lN,start_img,end_img)
                color = compute_color(pitch,volume,t,N,noteCounter,lN,start_img,end_img)
                color = (255-color[0],color[1],color[2])
                x0,y0 = dx+int(pitchScaled*breite),dy+int(volumeScaled*hoehe)
                subdiv.insert((x0,y0))
                img = draw_voronoi(img,subdiv,color)
                noteCounter += 1
        # draw circles    
        noteCounter = 0
        for tt in notes:
            note,start_img,end_img = tt
            start,pitches, duration, volume,rest = note
            if rest==1:
                continue
            volumeScaled = (volume-mv)/dv
            #print(t,note)        
            for pitch in pitches:
                pitchScaled = (pitch-mp)/dp
                radius = compute_radius(pitch,volume,t,N,noteCounter,lN,start_img,end_img)
                color = compute_color(pitch,volume,t,N,noteCounter,lN,start_img,end_img)
                #print(pitchScaled)
                x0,y0 = dx+int(pitchScaled*breite),dy+int(volumeScaled*hoehe)
                #subdiv.insert((x0,y0))
                img = draw_circle(img,pp=(x0,y0),color=(255-color[0],255-color[1],255-color[2]),radius=radius)    
                ## symmetric objects:
                ##m,k=3,2
                ##ff = FF(nn=[(volume//64)*m+k,(pitch//64)*m+k,2],m=m,k=k,aa=[(volume+pitch)/x*np.sin(t*np.pi*2/N)/2.0 for x in [128,128,128]])
                ##img = draw_curve(img,ff,(x0,y0),color,rr=2*radius,number_of_points = 1000)
                noteCounter += 1
            
        #labeled_img = color_img(img)
        #imgs.append(labeled_img)
        video.write(img)
    video.release()    
    return True

def kernPause(a1,a2):
    return  1*(a1==a2)

def kernPitch(k1,k2):
    q = getRational(k2-k1)
    a,b = q.numerator(),q.denominator()
    return gcd(a,b)**2/(a*b)

def kernDuration(k1,k2):
    return  min(k1,k2)/max(k1,k2)

def kernVolume(k1,k2):
    return min(k1,k2)/max(k1,k2)

def getRational(k):
    alpha = 2**(1/12.0)
    x = RDF(alpha**k).n(50)
    return x.nearby_rational(max_error=0.01*x)


def getCoordinatesOfPitchList(pitchList,dx,breite):
    M0 = matrix([[kernPitch(t1,t2) for t1 in pitchList] for t2 in pitchList],ring=RDF)
    if not M0.is_positive_definite():
        M0+= matrix.identity(len(pitchSet))*0.1
        
    from sklearn.decomposition import PCA
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import MinMaxScaler
    stdScaler = MinMaxScaler(feature_range=(dx,breite-dx))
    KPCA = KernelPCA(2,kernel='precomputed')
    
    Ch0 = KPCA.fit_transform(np.array(M0))
    
    X0 = [x for x in 1.0*Ch0]
    
    print(X0)
    
    X0 = stdScaler.fit_transform(X0)
    
    print(X0)
    
    invPitchDict = dict(zip(pitchList,range(len(pitchList))))
    return X0, invPitchDict

def make_video_with_symmetry(imgs2Notes,pitchSet,volumeSet,videoname="./opencv_videos/video.avi",fps=25,verbose=False):
    fourcc = cv.VideoWriter_fourcc(*"X264") 
    height,width = 512,512
    print(width,height,fps,videoname)
    framesPerSecond = fps
    video = cv.VideoWriter(videoname, fourcc, framesPerSecond, (width, height))
    cnt = 0

    print("volumeSet = ",volumeSet) 
    print("pitchSet = ", pitchSet)
    dx = 50
    dy = 10
    breite = 512-2*dx
    hoehe = 512-2*dy
    pitchList = sorted(list(pitchSet))
    
    X0,invPitchDict = getCoordinatesOfPitchList(pitchList,dx,breite)
    
    N = len(imgs2Notes.keys())
    import random
    mv = min(volumeSet)
    Mv = max(volumeSet)
    dv = Mv-mv
    if dv ==0:
        dv = 1
    mp = min(pitchSet)
    Mp = max(pitchSet)        
    dp = Mp-mp
    if dp ==0:
        dp = 1        

    #img = np.ones((512,512,3), np.uint8)*255 # white background
    rb0 = np.random.randint(breite//2,breite)
    rh0 = np.random.randint(hoehe//2,hoehe)
    
    
    Npoints = len(pitchSet)*5
    
    for t in range(N):
        img = np.ones((512,512,3), np.uint8)*255 # white background
        
        try:
            notes = imgs2Notes[t]
        except:
            continue    
        print(t,"/",N," img")
        m,k=3,2
        ln = len(notes)
        ps = []
        vs = []
        ds = []
        ts = []
        size = img.shape
        rect = (0,0,size[1],size[0])
        subdiv = cv.Subdiv2D(rect)
        points = []
        for tt in notes:
            note,start_img,end_img = tt
            tScaled = (t-start_img+1)/(end_img-start_img+1)
            start,pitches, duration, volume,rest = note
            print(note)
            pitchesScaled  = []
            for p in pitches:          
                pScaled = min(max(0,int((p-mp)/dp*Npoints)),Npoints-1)
                pitchesScaled.append(p)
                points.append([int(x) for x in X0[invPitchDict[p]].tolist()])
            ps.append(pitchesScaled)    
            vs.append(volume)
            ds.append(duration)
            ts.append(tScaled)
        print("ts = ", ts)
        print("points = ", points)  
        print("ps = ", ps)  
        nn = [(i)*m+k for i in range(4)]    
        aa = [((i+1)/4.0) for i in range(4)]
        print("nn = ", nn)
        print("aa = ", aa)
        #ff = FF(nn=nn,m=m,k=k,aa=aa)
        #img,points = draw_curve(img,ff,(256,256),(0,0,0),rr=int(30*np.median(vs)/64),number_of_points = Npoints,return_points=True)
        
        for pp in X0:
            print(pp)
            subdiv.insert(tuple([int(x) for x in pp]))
        img = draw_voronoi(img,subdiv,(128,32,255))            
        c = 0
        for i in range(ln):
            pitchesScaled = ps[i]
            for p in pitchesScaled:
                radius = max(10,int((vs[i]+ts[i])/4))
                color = (int(ts[i]*2*np.sin(2*np.pi*t/N)),int(ts[i]*vs[i]*2*np.sin(2*np.pi*t/N)),int(ts[i]*vs[i]*2*np.cos(2*np.pi*t/N)*128))
                img = draw_circle(img,pp=points[c],color=color,radius=radius) 
                c += 1

        #img = color_img(img)
        video.write(img)
    video.release()    
    return True


def make_video(imgs2Notes,pitchSet,volumeSet,videoname="./opencv_videos/video.avi",fps=25,video_type="voronoi",verbose=False):
    if video_type=="circle": return make_video_with_circles(imgs2Notes,pitchSet,volumeSet,videoname,fps,verbose)
    if video_type=="voronoi": return make_video_with_voronoi(imgs2Notes,pitchSet,volumeSet,videoname,fps,verbose)
    if video_type=="symmetry": return make_video_with_symmetry(imgs2Notes,pitchSet,volumeSet,videoname,fps,verbose)
    
import sys

print(sys.argv)

midi = sys.argv[1] #"./mix_of_midis/midi/markov-3_120bpm_1min_muttertag.mid"
scores = parseXml(fp=midi)            
fps = int(sys.argv[2])
bpm = int(sys.argv[3])
print("fps = ", fps, "bpm = ", bpm)
imgs2Notes,pitchSet,volumeSet = convertScore(scores,bpm=bpm,fps=fps,verbose=False)
#print(volumeSet)
#print(imgs2Notes)
vn = sys.argv[4]
video_type = sys.argv[5]
make_video(imgs2Notes,pitchSet,volumeSet,videoname=vn,fps=fps,video_type=video_type,verbose=True)    

