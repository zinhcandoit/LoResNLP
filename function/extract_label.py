import json

def parse_label_line(line):
    img_path, json_part = line.strip().split("\t", 1)
    boxes = json.loads(json_part)
    return img_path, boxes