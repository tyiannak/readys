# USAGE:
# convertToWav <folder path> <sampling rate> <number of channels>
#

import glob, sys, os

def getVideoFilesFromFolder(dirPath):
	types = (dirPath+os.sep+'*.avi', dirPath+os.sep+'*.mkv', dirPath+os.sep+'*.mp4',
			 dirPath+os.sep+'*.mp3', dirPath+os.sep+'*.flac', dirPath+os.sep+'*.ogg', dirPath+os.sep+'*.wav') # the tuple of file types
	files_grabbed = []
	for files in types:
		files_grabbed.extend(glob.glob(files))
	return files_grabbed

def main(argv):
	if (len(argv)==5):
		files = getVideoFilesFromFolder(argv[1])
		samplingRate = int(argv[2])
		channels = int(argv[3])
		out_path = argv[4]

		if not os.path.exists(out_path):
			os.makedirs(out_path)

		for f in files:
			new_path = '\"' + os.path.join(out_path, os.path.split(f)[-1].replace(" ","_"))+'.wav' + '\"'
			ffmpegString = 'ffmpeg -i ' + '\"' + f + '\"' + ' -ar ' + \
						   str(samplingRate) + ' -ac ' + str(channels) + \
						   ' ' + new_path
#			print(ffmpegString)
			os.system(ffmpegString)

if __name__ == '__main__':
	main(sys.argv)
