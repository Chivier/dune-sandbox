import re
import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
txt = open(os.path.join(BASE_DIR, "HIPAA.graphml"), "r", encoding="utf-8").read()

pattern = r'<node id="(.*)">\s*<data key="d0">(.*)<\/data>\n<\/node>'

matches = re.findall(pattern, txt)

with open(os.path.join(BASE_DIR, "HIPAA_extracted.txt"), "w", encoding="utf-8") as f:
    for match in matches:
        f.write(f"{match[0]}\t")
        f.write(f"{match[1]}\n")
