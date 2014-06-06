from modules.io.InputStream import VideoFileIS



vidPath = "../dataset/videos/GOPR0809_start_0_27_end_1_55.mp4"

if __name__ == '__main__':
	a = VideoFileIS(vidPath)
	a.getFrame()
	a.showVideo()