import json


def write_json(entry, filename,username): 
    with open(filename) as json_file: 
        data = json.load(json_file) 
    temp = data
    temp[username] = entry
    with open(filename,'w') as f: 
        json.dump(data, f, indent=4)