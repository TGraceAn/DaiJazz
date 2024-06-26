import regex as re
import os

def augmentation(path):
    # Load the txt file
    if not os.path.exists('txt_aug'):
        os.makedirs('txt_aug', exist_ok=True)

    for files in os.listdir(path):
        if files.endswith('.txt'):
            file_name = files[0:-4]
            for i in range(-5, 16):
                change_num(f'{path}/{files}', file_name, i)


###
# Only use when creating training data for conditional generation

def piano_jazz_aug(path):
    # Load the txt file
    if not os.path.exists('txt_aug'):
        os.makedirs('txt_aug', exist_ok=True)

    for files in os.listdir(path):
        if files.endswith('.txt'):
            file_name = files[0:-4]
            for i in range(-22, 23):
                change_num(f'{path}/{files}', file_name, i)


"""
Overall implementation of the augmentation function
"""

def change_num(path, file_name, num):
    with open(path, 'r') as file:
        data = file.readlines()

    updated_data = []
    track4sub = False
    for line in data:
        track_start = re.search(r'\[TRACK_START\]', line)
        if track_start:
            track4sub = True
            
        precussion = re.search(r'\[INSTRUMENT\] \d+\.', line)
        if precussion:
            track4sub = False
            
        if track4sub == True:
            note_represent = re.findall(r'(\d+\_\d+)', line)
            if note_represent:
                note = re.findall(r'\d+\_(\d+)', note_represent[0])

                if int(note_represent[0][0]) == 0 or int(note_represent[0][0]) == 1:
                    note = str(int(note[0]) + num)

                    if int(note) > 127:
                        note = str(127)
                    elif int(note) < 0:
                        note = str(0)

                    line = re.sub(r'\d+_\d+', f'{note_represent[0][0]}_{note}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_{num}.txt', 'w+') as file:
        file.writelines(updated_data)
    