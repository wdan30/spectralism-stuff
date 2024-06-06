import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft
from scipy.io import wavfile
from scipy.signal import find_peaks, convolve

def normalizeList(inputList, listMax, doRound: bool):
    for i in range(len(inputList)):
        inputList[i] /= max(inputList) / listMax
        if doRound:
            inputList[i] = round(inputList[i])
    
    return inputList

def doConvolution(inputList):
    output = []
    
    for elem in inputList:
        convolvedList = np.ndarray.tolist(convolve(elem[0], elem[1]))
        output.append(convolvedList)

    totalMax = np.amax(output)

    for i in range(len(output)):
        for j in range(len(output[i])):
            output[i][j] /= totalMax / 12
            output[i][j] = round(output[i][j])
        
    return output

def num_to_note(num, start):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    startIndex = notes.index(start)
    return notes[(num + startIndex) % 12]

def num_list_to_note(numList: list, start):
    output = []

    for num in numList:
        output.append(num_to_note(num, start))

    return output

def freq_to_note(freq):
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    note_number = 12 * np.log2(freq / 440) + 49  
    note_number = round(note_number)
        
    note = (note_number - 1 ) % len(notes)
    note = notes[note]
    
    octave = (note_number + 8 ) // len(notes)
    
    return note, octave

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

def generate_wave(type, freq, sample_rate, duration):
    pass

def fourier_transform(inputFile, upperBound):
    _, data = wavfile.read(inputFile)

    data = np.transpose(data)
    data1 = data[0]
    data1 = np.int16((data1 / max(data1)) * 32767)

    yf1 = fft(data1)
    yf1Abs = np.abs(yf1)

    xf = range(len(yf1))

    return(xf, yf1Abs)

    plt.plot(xf, yf1Abs)
    plt.xlim([0, upperBound])
    plt.show()

def ring_modulation(wave1, wave2):
    return np.multiply(wave1, wave2)

def get_peaks(inputArray, upperBound):
    peaks, _ = find_peaks(inputArray[27:upperBound], 10750000, None, 20)
    
    return peaks

inputList = [([3, 0, 9], [5, 0, 1, 12]),
             ([3, 0, 9], [5, 0, 3, 6]),
             ([1, 0, 7], [3, 2, 1, 2]),  
             ([0, 1, 7], [3, 2, 1, 0])]

convList = doConvolution(inputList)

print(convList)

print(num_list_to_note(convList[0], 'F'))
print(num_list_to_note(convList[1], 'C#'))
print(num_list_to_note(convList[2], 'A'))
print(num_list_to_note(convList[3], 'C#'))

convRhythm = np.ndarray.tolist(convolve([2, 2, 3], [2, 2, 2, 3]))

convRhythm = normalizeList(convRhythm, 3, True)

print(convRhythm)

fileName = "DeepBellRingHalf.wav"
x, yf1Abs = fourier_transform(fileName, 8372)

peaks = get_peaks(yf1Abs, 8372)

print(fileName)
for freq in peaks:
    print(f"{freq_to_note(freq)[0]}{freq_to_note(freq)[1]} {freq} {100 * yf1Abs[freq] / max(yf1Abs)}%")

plt.plot(x, yf1Abs)
plt.xlim([27, 4186])
plt.show()

upperBound = 8372
peaksAmplitude = []
duration = 5
sampleRate, data = wavfile.read('TubularBellsRing.wav')

sampleNum = len(data)

data = np.transpose(data)
data1 = data[0]
data1 = np.int16((data1 / data1.max()) * 32767)

yf1 = fft(data1)
yf1Abs = np.abs(yf1)

xf = range(len(yf1))

peaks, properties = find_peaks(yf1Abs[:upperBound], 12500000, None, 100)

waveform = np.linspace(0, duration, 41000 * duration, endpoint = False)

for index in peaks:
    peaksAmplitude.append(yf1Abs[index])

for i in range(len(peaks)):
    _, sineWave = generate_sine_wave(peaks[i], 41000,  5)
    sineWave = sineWave * (peaksAmplitude[i] / max(peaksAmplitude))
    waveform += sineWave

    noteName, noteOctave = freq_to_note(peaks[i])
    print(f"{peaks[i]}Hz, {noteName}{noteOctave}, {peaksAmplitude[i] * 100 / max(peaksAmplitude)}%")

wavfile.write("output.wav", 41000, waveform)

plt.plot(xf, yf1Abs)
plt.xlim([0, upperBound])
plt.show()
