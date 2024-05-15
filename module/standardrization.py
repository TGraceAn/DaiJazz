from mido import MidiFile, midifiles
import os

"""
This is use to standarize the data into type 1 format
"""

"""
preprocess the type 1 data

"""
def pre_process_type_1(mid, file_name):
    #Save into a text file which is easy to read
    #Make a new directory to save the text file
    if not os.path.exists('txt_out'):
        os.makedirs('txt_out', exist_ok=True)

    with open(f'txt_out/{file_name}.txt', 'w+') as file:
        for i, track in enumerate(mid.tracks):
            file.write('Track {}: {}\n'.format(i, track.name))
            for msg in track:
                file.write(str(msg) + '\n')

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

    for msg in mid.tracks[0]:
        total_time += msg.time
        if msg.type == 'sysex':
            sysex_message = msg
        elif isinstance(msg, midifiles.meta.MetaMessage) and msg.type == 'end_of_track':
            end_of_track_msg = msg
        elif isinstance(msg, midifiles.meta.MetaMessage):
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
    end_track.append(sysex_message)
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
    #Save into a txt file which is easy to read
    #Make a new directory to save the txt file
    if not os.path.exists('txt_out'):
        os.makedirs('txt_out', exist_ok=True)

    with open(f'txt_out/{file_name}.txt', 'w+') as file:
        for i, track in enumerate(final_track):
            file.write('Track {}: {}\n'.format(i, track[0].name))
            for msg in track:
                file.write(str(msg) + '\n')
        
def standardrize(path):
    for files in os.listdir(path):
        file = os.path.join(path, files)
        if files.endswith('.mid'):
            file_name = files[0:-4]
            mid = MidiFile(file)
            # print(f'{file_name}')
            if mid.type == 0:
                # print('Type 0')
                change021(mid, file_name)
            elif mid.type == 1:
                # print('Type 1')
                pre_process_type_1(mid, file_name)
            elif mid.type == 2:
                # print(f'Ignoring {files} as it is type 2')
                pass
        elif files.endswith('.midi'):
            file_name = files[0:-5]
            mid = MidiFile(file)
            if mid.type == 0:
                change021(mid, file_name)
            elif mid.type == 1:
                pre_process_type_1(mid, file_name)
            elif mid.type == 2:
                # print(f'Ignoring {files} as it is type 2')
                pass


"""
TODO: 
Write a reversed translation code from txt files to midi files

Read Jazz Transformer
Do encoder stuff
"""

if __name__ == '__main__':
    path = 'midi_files'
    standardrize(path)
