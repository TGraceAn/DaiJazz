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
            
        precussion = re.search(r'\[INSTRUMENT\]\:\d+\.', line)
        if precussion:
            track4sub = False
            
        if track4sub == True:
            note_represent = re.search(r'(\[WHAT\]\:\d+ \[WHICH\]\:\d+ \[HOW\]\:\d+ \[WHEN\]\:\d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0][-1]) == 0 or int(note_represent[0][-1]) == 1:
                    note = re.findall(r'\[WHICH\]\:(\d+)',line)
                    
                    note = str(int(note[0]) + num)

                    if int(note) > 127:
                        note = str(127)
                    elif int(note) < 0:
                        note = str(0)

                    line = re.sub(r'\[WHICH\]\:\d+', f'[WHICH]:{note}', line)

                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_{num}.txt', 'w+') as file:
        file.writelines(updated_data)
    