from mido import MidiFile, midifiles, MetaMessage, MidiTrack
import os
from data_analysis import max_resolution

# standard = max_resolution('data')

standard = 1024


def pre_process_type_1(mid, file_name):
    playing = False
    resolution = mid.ticks_per_beat
    
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

                    time = msg.time/resolution
                    time = time*standard
                    if int(time + 0.5) == int(time) + 1:
                        msg.time = int(time + 0.5)
                    else:
                        msg.time = int(time)

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
                                    message = f'[INSTRUMENT]:{msg.program}.'
                                    non_added_time += msg.time
                                    adding = True
                                    file.write(message + '\n')
                            else:
                                if msg.type == 'program_change':
                                    message = f'[INSTRUMENT]:{msg.program}'
                                    non_added_time += msg.time
                                    adding = True
                                    file.write(message + '\n')

                            if msg.type == 'note_on':
                                message = '[WHAT]:0_{} [HOW]:{} [WHEN]:{}'.format(msg.note, msg.velocity, msg.time) #range if velocity [0,127]
                                file.write(message + '\n')
                            elif msg.type == 'note_off':
                                msg.velocity = 0
                                message = '[WHAT]:0_{} [HOW]:{} [WHEN]:{}'.format(msg.note, msg.velocity, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'control_change':
                                message = '[WHAT]:2_{} [CC_V]:{} [WHEN]:{}'.format(msg.control, msg.value, msg.time)
                                file.write(message + '\n')
                            elif msg.type == 'polytouch':
                                message = '[WHAT]:3_{} [PT_V]:{} [WHEN]:{}'.format(msg.note, msg.value, msg.time)
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

                    time = msg.time/resolution
                    time = time*standard
                    if int(time + 0.5) == int(time) + 1:
                        msg.time = int(time + 0.5)
                    else:
                        msg.time = int(time)

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
                                    message = f'[INSTRUMENT]:{msg.program}.'
                                    non_added_time += msg.time
                                    adding = True
                                    file.write(message + '\n')
                            else:
                                if msg.type == 'program_change':
                                    message = f'[INSTRUMENT]:{msg.program}'
                                    non_added_time += msg.time
                                    adding = True
                                    file.write(message + '\n')

                            if msg.type == 'note_on':
                                message = '[WHAT]:0_{} [HOW]:{} [WHEN]:{}'.format(msg.note, msg.velocity, msg.time) #range if velocity [0,127]
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