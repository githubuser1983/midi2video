./mid2vid.sh ./Birthday.mid 100 120 ./video_Birthday.avi voronoi
./Midi2mp3.sh /usr/share/sounds/sf2/SGM-v2.01-CompactGrand-Guit-Bass-v2.7.sf2 ./Birthday.mid
./mergeAudioAndVideoToVideo.sh Birthday.mp3 video_Birthday.avi video_Birthday.mp4
