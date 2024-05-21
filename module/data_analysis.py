import regex as re
import os

def analyze_data(path):
    max_note = find_highestnote(path)
    min_note = find_lowestnote(path)
    
    return max_note, min_note
                            

def find_highestnote(path):
    max_note = 0
    for files in os.listdir(path):
        if files.endswith('+6.txt'):
            file_name = files[0:-6]
            with open(f'{path}/{file_name}+6.txt', 'r') as f:
                data = f.readlines()
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
                            if int(note_represent[0]) == 0:
                                if int(note_represent[1]) > max_note:
                                    max_note = int(note_represent[1])
    return max_note

def find_lowestnote(path):
    min_note = 127
    for files in os.listdir(path):
        if files.endswith('-5.txt'):
            file_name = files[0:-6]
            with open(f'{path}/{file_name}-5.txt', 'r') as f:
                data = f.readlines()
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
                            if int(note_represent[0]) == 0:
                                if int(note_represent[1]) < min_note:
                                    if int(note_represent[1]) == 0:
                                        min_note = int(note_represent[1])
                                
    return min_note

def find_avgnote():
    # Code to find the average note in the data
    pass

def find_medianote():
    # Code to find the median note in the data
    pass