# $1 audio.mp3
# $2 video.avi
# $3 merged.mp4
# Beispiel:
# ./imgAudioToVideo.sh image.jpg audio.mp3 video.mp4
ffmpeg  -i $1 -i $2 -shortest -c copy $3
