import glob
import pickle
from music21 import *

import midi_util

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    chords = []

    for file in glob.glob("joe/*.mid"):
        midi_file = converter.parse(file)

        part1 = midi_file.parts[0]
        part2 = midi_file.parts[1]

        print("Parsing %s" % file)
        # part1.show('text')
        notes_to_parse = None

        notes_to_parse = part1.recurse()
        chord_to_parse = part2.recurse()

        for element in notes_to_parse:

            offset = element.offset

            if isinstance(element, note.Note):
                print(offset, element.pitch, element.duration)
                notes.append(str(element))
            elif isinstance(element, chord.Chord):
                print(offset, element.pitchNames, element.duration)
                notes.append('.'.join(str(n) for n in element.normalOrder))
            elif isinstance(element, note.Rest):
                print(offset, element, element.duration)
            else:
                print(element)
                pass

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    stream_player = midi.realtime.StreamPlayer(midi_file)
    stream_player.play()

    return notes

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
    midi_file = converter.parse("MyOwnDataset/Arriettys_Song_The_Secret_World_of_Arrietty.mid")
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
    midi_file = converter.parse("MyOwnDataset/Vague_Hope__Cold_Rain_NieRAutomata.mid")
    
    pitchesTable = [pitch.Pitch(ps) for ps in range(17, 88)]
    print(len(pitchesTable))

    melody = midi_file.parts[0]
    accomp = midi_file.parts[1]

    for measure in melody.measures(0, None, collect="Measure"):
        print(measure)
        notes_to_parse = measure.recurse().notesAndRests
        offsetIter = stream.iterator.OffsetIterator(notes_to_parse)
        for elementGroup in offsetIter:
            element = elementGroup[0]
            
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
    analysis_by_measures()

    # print(midi_file.parts[0].show("text"))
    # print(midi_file.parts[0].getInstrument())
    # for part in s2.parts:
    #     part_instroment = part.getInstrument()
    #     if isinstance(part_instroment, instrument.Piano):
    #         part.show("text")
    # get_notes()

