import glob
import pickle
from music21 import *

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
    midi_file = converter.parse("MyOwnDataset/Angel_Beats_Unjust_Life.mid")
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
    midi_file = converter.parse("myown/Stay_Alive__ReZero_ED_2__TheIshter_Sheet_Music__Full_Sheets.mid")
    
    # pitchesTable = [pitch.Pitch(ps) for ps in range(17, 88)]
    # print(len(pitchesTable))
    
    melody = midi_file.parts[0]
    # accomp = midi_file.parts[1]

    for measure in melody.measures(0, None, collect="Measure"):
        print(measure)

        notes_to_parse = measure.recurse().notesAndRests
        offset_iter = stream.iterator.OffsetIterator(notes_to_parse)

        for element_group in offset_iter:
            element = element_group[0]
            
            if isinstance(element, note.Rest):
                ps = 0
            elif isinstance(element, note.Note):
                ps = element.pitch.ps
            elif isinstance(element, chord.Chord):
                ps = element.pitches[0].ps

            print("offset:{} pitch space:{} duarion:{}".format(element.offset, ps, element.duration.quarterLength))


    # print(type(midi_file.parts[0]))

    # cr = analysis.reduceChords.ChordReducer()
    # cws = cr.computeMeasureChordWeights()
    # for pcs in sorted(cws):
    #     print("%18r  %2.1f" % (pcs, cws[pcs]))

    stream_player = midi.realtime.StreamPlayer(midi_file)
    stream_player.play()

if __name__ == '__main__':
    # analysis_by_measures()
    print(note.Note(KeyToPitch[87]).pitch.ps)
    # midi_path = "midi_songs/*.mid"
    # data_path = "midi_input/data"
    # data = midi_util.parse_midi(midi_path, data_path)


