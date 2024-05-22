from mido import MidiFile, midifiles, MetaMessage, MidiTrack
import os

"""
TOKENS Explanations

###
MESSAGE: The message that is being sent

### WHAT?
NOTE_ON: Specify the beginning of the note (0)
NOTE_OFF: Specify the end of the note (Sending a note_on with velocity = 0 is considered a note_off) (1)
CONTROL_CHANGE: For uses in pedals and stuffs (2)

### WHICH?
NOTE: Which note do I play
CONTROL: Which control am I using (NOOOOOO NONE OF THE DATASETS HAS THIS)

### HOW?
VELOCITY: How hard the note is played
VALUE: How should I use it? (NOOOOOO NONE OF THE DATASETS HAS THIS)

### WHEN?
DELTA_TIME: Specify when it starts after the previous message

PIECE_START: DENOTE THE START OF THE PIECE
TRACK_START: DENOTE THE START OF A TRACK IN A PIECE
TRACK_END: DENOTE THE END OF A TRACK IN A PIECE
BAR_START:
BAR_END:
TIME_SHIFT: Can use in place of DELTA_TIME

INSTRUMENT: Specify the instrument of choice

FILL_PLACEHOLDER: Notify the model where to fill in the blanks

FILL_START: 
FILL_END: 
"""


TOKENS = ['[MESSAGE]','[WHAT]','[WHICH]', '[HOW]',
          '[WHERE]','[DELTA_TIME]','[PIECE_START]',
          '[TRACK_START]', '[TRACK_END]','[BAR_START]',
          '[BAR_END]','[TIME_SHIFT]', '[INSTRUMENT]',
          '[FILL_PLACEHOLDER]','[FILL_START]', '[FILL_END]']

ADDED_TOKENS = ['[PIECE_START]','[TRACK_START]', '[TRACK_END]',
                '[INSTRUMENT]','[FILL_PLACEHOLDER]','[FILL_START]', '[FILL_END]']


"""
preprocess the type 1 data
"""

message_dict = {
    'note_on': 0,
    'note_off': 1,
    'control_change': 2,
    'polytouch': 3,
    'pitchwheel': 4,
    'aftertouch': 5,
}
def pre_process_type_1(mid, file_name):
    playing = False
    
    #Save into a text file which is easy to read
    #Make a new directory to save the text file
    if not os.path.exists('txt_out'):
        os.makedirs('txt_out', exist_ok=True)

    with open(f'txt_out/{file_name}.txt', 'w+') as file:
        #Add the piece start token
        file.write('[PIECE_START]\n')
        for i, track in enumerate(mid.tracks):
            playing = False
            
            for msg in track:
                if msg.type == "note_on":
                    playing = True
                    break
                    
            if playing == True:
                file.write('[TRACK_START]\n')
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
                        # file.write(str(msg) + '\n')

                    else:
                        try:
                            if msg.channel == 9: #If the channel is 9, it is a percussion instrument
                                if msg.type == 'program_change':
                                    message = f'[INSTRUMENT] {msg.program}.'
                                    file.write(message + '\n')
                            else:
                                if msg.type == 'program_change':
                                    message = f'[INSTRUMENT] {msg.program}'
                                    file.write(message + '\n')

                            if msg.type == 'note_on':
                                message = '0 {} {} {}'.format(msg.note, msg.velocity, msg.time)
                                if msg.velocity == 0:
                                    message = '1 {} {} {}'.format(msg.note, msg.velocity, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'note_off':
                                msg.velocity = 0
                                message = '1 {} {} {}'.format(msg.note, msg.velocity, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'control_change':
                                message = '2 {} {} {}'.format(msg.control, msg.value, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'polytouch':
                                message = '3 {} {} {}'.format(msg.note, msg.value, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'pitchwheel':
                                message = '4 {} {}'.format(msg.pitch, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'aftertouch':
                                message = '5 {} {}'.format(msg.value, msg.time)
                                file.write(message + '\n')
                        except:
                            non_added_time += msg.time
                            adding = True
                            # file.write(str(msg) + '\n')

            #Add the track end token
            file.write('[TRACK_END]\n')


"""
process the type 0 data to txt file similar to type 1 and saved a type 1 version of the midifile
"""

def change021_txt(mid, file_name):
    # Variables
    total_time = 0
    num_channel = 0 #Number of tracks = number of channels + header track and end of track
    channel_list = [] #List of existing channels
    list_channel = [] #List of messages in each channel
    track_delta_time = [] #To sync the time of each channel
    time_stemp = 0 #Initial time stemp for all the tracks to be sync
    header_track = []
    end_track = []
    final_track = []
    sysex_message = None
    end_of_track_msg = None
    resolution = mid.ticks_per_beat

    for msg in mid.tracks[0]:
        total_time += msg.time
        if msg.type == 'sysex':
            sysex_message = msg
        elif isinstance(msg, midifiles.meta.MetaMessage) and msg.type == 'end_of_track':
            end_of_track_msg = msg
        elif isinstance(msg, midifiles.meta.MetaMessage):
            if msg.type == 'copyright':
                time_stemp += msg.time
            else:
                header_track.append(msg)
                time_stemp += msg.time
        elif msg.channel not in channel_list:
            time_stemp += msg.time
            channel_list.append(msg.channel)
            # # Update delta time for all channels
            # First control ever
            if len(track_delta_time) == 0:
                msg.time = time_stemp
                track_delta_time.append(msg.time)
            else:
                for i in range(len(track_delta_time)):
                    track_delta_time[i] += msg.time 
                    track_delta_time.append(msg.time)
            msg.time = time_stemp
            list_channel.append([msg])
            if len(track_delta_time) == 1:
                track_delta_time[0] = 0
            else:
                track_delta_time[-1] = 0
            num_channel += 1
        elif msg.channel in channel_list: 
            time_stemp += msg.time
            # If the same channel then update delta time, else add more time to delta time
            for i, channel in enumerate(channel_list):
                track_delta_time[i] += msg.time
            #If message is the same channel as delta time, add message to the corresponding channel and return its delta time to 0
            for i, channel in enumerate(channel_list):
                if msg.channel == channel:
                    msg.time = track_delta_time[i]
                    list_channel[i].append(msg)
                    track_delta_time[i] = 0
                    
    #Add end of track message to each channel
    for i in range(num_channel):
        list_channel[i].append(end_of_track_msg)
        
#     if sysex_message:
#         end_track.append(sysex_message)
        
    end_track.append(end_of_track_msg)
    header_track.append(midifiles.meta.MetaMessage('end_of_track', time = total_time + 1000))
    final_track = [header_track]
    final_track += list_channel
    final_track += [end_track]
    #Add name to each track
    for i, track in enumerate(final_track):
        if i == 0:
            #Add to first of the track
            final_track[i].insert(0, midifiles.meta.MetaMessage('track_name', name = 'Header Track'))
        elif i == num_channel + 1:
            final_track[i].insert(0, midifiles.meta.MetaMessage('track_name', name = 'End of Track'))
        else:
            final_track[i].insert(0, midifiles.meta.MetaMessage('track_name', name = 'Channel {}'.format(i)))

    #Create type 1 midi file
    mid = MidiFile(ticks_per_beat = resolution)
    mid.tracks.append(final_track[0])

    for track in final_track:
        playing = False
        track = MidiTrack(track)
        for msg in track:
            if msg.type == 'note_on':
                playing = True
                break
        if playing == True:
            mid.tracks.append(track)
    
    mid.tracks.append(final_track[-1])

    #Save the midi file
    if not os.path.exists('midi_files'):
        os.makedirs('midi_files', exist_ok=True)    
    try:
        mid.save(f'midi_files/{file_name}_type1.mid')
    except:
        print(f'Could not save {file_name}_type1.mid')
    
        
    #Save into a txt file which is easy to read
    #Make a new directory to save the txt file
    if not os.path.exists('txt_out'):
        os.makedirs('txt_out', exist_ok=True)

    with open(f'txt_out/{file_name}.txt', 'w+') as file:
        #Add the piece start token
        file.write('[PIECE_START]\n')
        for i, track in enumerate(mid.tracks):
            playing = False
            for msg in track:
                if msg.type == "note_on":
                    playing = True
                    break
            
            if playing == True:
                file.write('[TRACK_START]\n')
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
                        # file.write(str(msg) + '\n')

                    else:
                        try:
                            if msg.channel == 9: #If the channel is 9, it is a percussion instrument
                                if msg.type == 'program_change':
                                    message = f'[INSTRUMENT] {msg.program}.'
                                    file.write(message + '\n')
                            else:
                                if msg.type == 'program_change':
                                    message = f'[INSTRUMENT] {msg.program}'
                                    file.write(message + '\n')

                            if msg.type == 'note_on':
                                message = '0 {} {} {}'.format(msg.note, msg.velocity, msg.time)
                                if msg.velocity == 0:
                                    message = '1 {} {} {}'.format(msg.note, msg.velocity, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'note_off':
                                msg.velocity = 0
                                message = '1 {} {} {}'.format(msg.note, msg.velocity, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'control_change':
                                message = '2 {} {} {}'.format(msg.control, msg.value, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'polytouch':
                                message = '3 {} {} {}'.format(msg.note, msg.value, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'pitchwheel':
                                message = '4 {} {}'.format(msg.pitch, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'aftertouch':
                                message = '5 {} {}'.format(msg.value, msg.time)
                                file.write(message + '\n')
                        except:
                            non_added_time += msg.time
                            adding = True
                            # file.write(str(msg) + '\n')

            #Add the track end token
            file.write('[TRACK_END]\n')
        
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
                
            if mid.type == 0:
                change021_txt(mid, file_name)
            elif mid.type == 1:
                pre_process_type_1(mid, file_name)
            elif mid.type == 2:
                pass
        elif files.endswith('.midi'):
            file_name = files[0:-5]

            try:
                mid = MidiFile(file)   
            except:
                continue
            else:
                pass

            if mid.type == 0:
                change021_txt(mid, file_name)
            elif mid.type == 1:
                pre_process_type_1(mid, file_name)
            elif mid.type == 2:
                pass
