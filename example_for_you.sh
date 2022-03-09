./mid2vid.sh ./For_You_cello.mid 100 80 ./video_For_You.avi circle
./Midi2mp3.sh /usr/share/sounds/sf2/SGM-v2.01-CompactGrand-Guit-Bass-v2.7.sf2 ./For_You_cello.mid
./mergeAudioAndVideoToVideo.sh For_You_cello.mp3 video_For_You.avi video_For_You.mp4
