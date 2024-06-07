from mido import MidiFile, midifiles, MetaMessage, MidiTrack
import os

def piano_standard(mid, file_name):
    
    #Save into a text file which is easy to read
    #Make a new directory to save the text file
    if not os.path.exists('txt_out'):
        os.makedirs('txt_out', exist_ok=True)

    with open(f'txt_out/{file_name}.txt', 'w+') as file:
        #Add the piece start token
        file.write('[PIECE_START]\n')
        for i, track in enumerate(mid.tracks):
            
            # file.write('Track {}: {}\n'.format(i, track.name))
            #Add the track start token
            
            non_added_time = 0
            adding = False

            for msg in track:
                if adding:
                    msg.time += non_added_time
                    non_added_time = 0
                    adding = False

                if isinstance(msg,MetaMessage):
                    non_added_time += msg.time
                    adding = True

                else:
                    try:
                        if msg.channel == 9: #If the channel is 9, it is a percussion instrument
                            if msg.type == 'program_change':
                                message = f'[INSTRUMENT]:{msg.program}.'
                                non_added_time += msg.time
                                adding = True
                        else:
                            if msg.type == 'program_change':
                                message = f'[INSTRUMENT]:{msg.program}'
                                non_added_time += msg.time
                                adding = True
                        


                        if msg.type == 'note_on':
                            # Add padding to the velocity for inference
                            if msg.velocity <= 4:
                                msg.velocity = 4
                            elif msg.velocity >= 127:
                                msg.velocity = 116

                            message = '[WHAT]:0_{} [HOW]:{} [WHEN]:{}'.format(msg.note, msg.velocity, msg.time)
                            file.write(message + '\n')
                        elif msg.type == 'note_off':
                            msg.velocity = 0
                            message = '[WHAT]:0_{} [HOW]:{} [WHEN]:{}'.format(msg.note, msg.velocity, msg.time)
                            file.write(message + '\n')
                        elif msg.type == 'control_change':
                            message = '[WHAT]:2_{} [CC_V]:{} [WHEN]:{}'.format(msg.control, msg.value, msg.time)
                            file.write(message + '\n')
                        elif msg.type == 'polytouch':
                            message = '[WHAT]:3_{} [HOW]:{} [WHEN]:{}'.format(msg.note, msg.value, msg.time)
                            file.write(message + '\n')
                        elif msg.type == 'pitchwheel':
                            message = '[WHAT]:4 [P_V]:{} [WHEN]:{}'.format(msg.pitch, msg.time)
                            file.write(message + '\n')
                        elif msg.type == 'aftertouch':
                            message = '[WHAT]:5 [AT_V]:{} [WHEN]:{}'.format(msg.value, msg.time)
                            file.write(message + '\n')
                    except:
                        non_added_time += msg.time
                        adding = True
                        # file.write(str(msg) + '\n')

        #Add the piece end token
        file.write('[PIECE_END]\n')


def standardrize(path):
    for files in os.listdir(path):
        file = os.path.join(path, files)
        if files.endswith('.mid'):
            file_name = files[0:-4]
            try:
                mid = MidiFile(file)   
            except:
                continue
            else:
                pass
                
            piano_standard(mid, file_name)

        elif files.endswith('.midi'):
            file_name = files[0:-5]

            try:
                mid = MidiFile(file)   
            except:
                continue
            else:
                pass

            piano_standard(mid, file_name)

if __name__ == '__main__':
    standardrize('midi_files')