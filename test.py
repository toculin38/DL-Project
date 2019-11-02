import glob
import pickle
from music21 import *
import numpy as np
import midi_util

KeyToPitch = dict((number, float(pitch)) for number, pitch in enumerate(range(21, 109)))
PitchTokey = dict((float(pitch), number) for number, pitch in enumerate(range(21, 109)))

def make_a_song():
    ts = meter.TimeSignature('6/8')
    mm1 = tempo.MetronomeMark(number=64.5)
    a = stream.Stream()
    a.append(mm1)
    a.append(ts)


    n = note.Rest()
    n.duration.quarterLength = 0.5
    a.append(n)

    n = chord.Chord(['A', 'C', 'F'])
    n.duration.quarterLength = 0.5
    a.append(n)

    n = note.Note('A4')
    n.duration.quarterLength = 0.5
    a.append(n)

    n = note.Rest()
    n.duration.quarterLength = 0.5
    a.append(n)

    n = chord.Chord(['A', 'C', 'F'])
    n.duration.quarterLength = 0.5
    a.append(n)

    n = note.Note('A4')
    n.duration.quarterLength = 0.5
    a.append(n)

    a.show('text')

    stream_player = midi.realtime.StreamPlayer(a)
    stream_player.play()

def analysis_test():
    midi_file = converter.parse("MyOwnDataset/Octopath_Traveler_-_Alfyns_Theme.mid")
    midi_file = midi_util.to_c_major(midi_file)

    print(len(midi_file.parts))
    print(midi_file.parts[0].getInstrument())
    print(midi_file.parts[1].getInstrument())

    # midi_file.show("text")
    for element in midi_file.parts[0].recurse():
        if isinstance(element, note.Note):
            # if element.offset % 0.25 == 0:
            print(element.pitch, element.duration)
        elif isinstance(element, chord.Chord):
            # if element.offset % 0.25 == 0:
            print(element.pitchNames, element.duration)
        elif isinstance(element, note.Rest):
            # if element.offset % 0.25 == 0:
            print("rest", element.duration)
        else:
            print(type(element))

    stream_player = midi.realtime.StreamPlayer(midi_file).play()

    midi_stream = stream.Stream(midi_file.parts[0])
    midi_stream.write('midi', fp='test.mid')

def analysis_by_measures():
    midi_file = converter.parse("midi_songs/4-4/Octopath_Traveler_-_Alfyns_Theme.mid")
    
    # pitchesTable = [pitch.Pitch(ps) for ps in range(17, 88)]
    # print(len(pitchesTable))
    
    melody = midi_file.parts[0].measures(0, None, collect="Measure")
    accomp = midi_file.parts[1].measures(0, None, collect="Measure")

    for measure in zip(melody, accomp):
        print(measure)
        melody_to_parse = measure[0].recurse().notesAndRests
        accomp_to_parse = measure[1].recurse().notesAndRests

        melody_offset_iter = stream.iterator.OffsetIterator(melody_to_parse)
        accomp_offset_iter = stream.iterator.OffsetIterator(accomp_to_parse)

        for element_group1, element_group2 in zip(melody_offset_iter, accomp_offset_iter):
            element1 = element_group1[0]
            element2 = element_group2[0]

            if isinstance(element1, note.Rest):
                ps = 0
            elif isinstance(element1, note.Note):
                ps = element1.pitch.ps
            elif isinstance(element1, chord.Chord):
                ps = element1.pitches[0].ps

            print("offset:{} pitch space:{} duarion:{}".format(element1.offset, ps, element1.duration.quarterLength))
            
            if isinstance(element2, note.Rest):
                ps = 0
            elif isinstance(element2, note.Note):
                ps = element2.pitch.ps
            elif isinstance(element2, chord.Chord):
                ps = element2.pitches[0].ps

            print("offset:{} pitch space:{} duarion:{}".format(element2.offset, ps, element2.duration.quarterLength))


    # print(type(midi_file.parts[0]))

    # cr = analysis.reduceChords.ChordReducer()
    # cws = cr.computeMeasureChordWeights()
    # for pcs in sorted(cws):
    #     print("%18r  %2.1f" % (pcs, cws[pcs]))

    stream_player = midi.realtime.StreamPlayer(midi_file)
    stream_player.play()



if __name__ == '__main__':
    # PressSize = 32
    # CycleLength = 128
    # OffsetBitSize = int(np.log2(CycleLength))

    # offset_pattern = np.array([[x] for x in range (64, 64 + PressSize)], dtype=np.uint8)
    # offset_pattern = np.unpackbits(offset_pattern, axis=-1)[:,-OffsetBitSize:]
    # print(offset_pattern)
    # new_offset_pattern = np.packbits(offset_pattern, axis=-1) // (2 ** (8 - OffsetBitSize))
    # new_offset_pattern = (new_offset_pattern + PressSize) % CycleLength
    
    # new_offset_pattern = np.unpackbits(new_offset_pattern, axis=-1)
    # offset_pattern = new_offset_pattern[:,-OffsetBitSize:]
    # print(offset_pattern)
    
    
    # new_offset_pattern = new_offset_patten % OffsetSize
    # offset_pattern = np.unpackbits(new_offset_pattern, axis=-1)


    analysis_by_measures()

    # midi_path = "midi_songs/*.mid"
    # data_path = "midi_input/data"
    # data = midi_util.parse_midi(midi_path, data_path)


