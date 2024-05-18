import regex as re
import os

def augmentation(path):
    # Load the txt file
    if not os.path.exists('txt_aug'):
        os.makedirs('txt_aug', exist_ok=True)

    for files in os.listdir(path):
        if files.endswith('.txt'):
            file_name = files[0:-4]
            change_1(f'{path}/{files}', file_name)
            change_2(f'{path}/{files}', file_name)
            change_3(f'{path}/{files}', file_name)
            change_4(f'{path}/{files}', file_name)
            change_5(f'{path}/{files}', file_name)
            change_6(f'{path}/{files}', file_name)
            change_7(f'{path}/{files}', file_name)
            change_8(f'{path}/{files}', file_name)
            change_9(f'{path}/{files}', file_name)
            change_10(f'{path}/{files}', file_name)
            change_11(f'{path}/{files}', file_name)

# Use regex to change the note in the note_on and note_off message by +1
### Data is represented in the following format: what which how when for example: 0 26 45 454

def change_1(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) + 1)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_+1.txt', 'w+') as file:
        file.writelines(updated_data)

def change_2(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) + 2)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)
            
    with open(f'txt_aug/{file_name}_+2.txt', 'w+') as file:
        file.writelines(updated_data)

def change_3(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) + 3)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_+3.txt', 'w+') as file:
        file.writelines(updated_data)

def change_4(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) + 4)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_+4.txt', 'w+') as file:
        file.writelines(updated_data)

def change_5(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) + 5)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_+5.txt', 'w+') as file:
        file.writelines(updated_data)

def change_6(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) + 6)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_+6.txt', 'w+') as file:
        file.writelines(updated_data)


def change_7(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) - 5)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_-5.txt', 'w+') as file:
        file.writelines(updated_data)

def change_8(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) - 4)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_-4.txt', 'w+') as file:
        file.writelines(updated_data)

def change_9(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) - 3)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_-3.txt', 'w+') as file:
        file.writelines(updated_data)

def change_10(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) - 2)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_-2.txt', 'w+') as file:
        file.writelines(updated_data)

def change_11(path, file_name):
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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) - 1)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_-1.txt', 'w+') as file:
        file.writelines(updated_data)

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
            note_represent = re.search(r'(\d+ \d+ \d+ \d+)', line)
            if note_represent:
                note_represent = note_represent.group(1)
                note_represent = note_represent.split(' ')

                if int(note_represent[0]) == 0 or int(note_represent[0]) == 1:
                    note_represent[1] = str(int(note_represent[1]) + num)
                    line = re.sub(r'\d+ \d+ \d+ \d+', f'{note_represent[0]} {note_represent[1]} {note_represent[2]} {note_represent[3]}', line)
                updated_data.append(line)
            else:
                updated_data.append(line)    
        else:
            updated_data.append(line)

    with open(f'txt_aug/{file_name}_{num}.txt', 'w+') as file:
        file.writelines(updated_data)
    