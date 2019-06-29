import glob
import pickle
from music21 import *

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
    midi_file = converter.parse("midi_songs/Kingdom_Hearts_Dearly_Beloved.mid", format='singleBeat')
    # midi_file = instrument.partitionByInstrument(midi_file)
    print(len(midi_file.parts))
    print(midi_file.parts[0].getInstrument())
    print(midi_file.parts[1].getInstrument())
    # midi_file.show("text")

    new_stream = stream.Stream()

    for element in midi_file.parts[0].flat:
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
            print(element)



    # elements = new_song.measures(0, None)
    # measures = elements.getElementsByClass('Measure')
    
    # for measure in measures:
    #     print(measure)
    #     for n in measure.flat.notes:
    #         print("Note:{} Beat:{}".format(n, n.duration))

    stream_player = midi.realtime.StreamPlayer(new_stream)
    stream_player.play()

    # midi_stream = stream.Stream(new_song)
    # midi_stream.write('midi', fp='test.mid')

if __name__ == '__main__':
    analysis_test()

    # print(midi_file.parts[0].show("text"))
    # print(midi_file.parts[0].getInstrument())
    # for part in s2.parts:
    #     part_instroment = part.getInstrument()
    #     if isinstance(part_instroment, instrument.Piano):
    #         part.show("text")
    # get_notes()

