
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse, glob, os, time
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('audio_path', type=str,
                    help='Path to the directory containing audio files.')
parser.add_argument('output_image_path', type=str,
                    help='Directory where the images will be saved in .png format.')


def audiotoimage(audio_path='../data/set_a/', output_image_path='../images/set_a_imgs/'):

    '''

    For each audio file in a folder, it creates an MFCC image
    '''

    dpi = 128

    for imag_a in glob.glob(audio_path + '*.wav'):
        for i in os.listdir(audio_path):
            # MFCC of normal heartbeat
            y, sr = librosa.load(imag_a)

            # visualize it
            fig = plt.Figure()
            # canvas = FigureCanvas(fig)
            D = np.abs(librosa.stft(y))
            plt.figure(figsize=(8, 4))
            ax1 = plt.subplot(2, 1, 1)
            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)
            librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log')
            plt.savefig(output_image_path + i.split('.')[0] + '.png', bbox_inches='tight', pad_inches=0, dpi=dpi)
            time.sleep(1)
            plt.close()

if __name__ == "__main__":
    audiotoimage('../data/set_b/', '../images/set_b_imgs/')
    print('All images are saved in folder.')
