
"""
 AUTHOR : Lang
 PURPOSE : Multi Self Deep Learning
"""

__author__ = 'Lang'

import re

wf = open('Names.txt','wt')
rf = open('NameAndFamilyWithTD.txt','rt')

pattern = re.compile(r'<a.+>')

species_count = 0
count = -10
line4 = ''

for line in rf:
    line1 = line.replace('<td>','')
    line2 = line1.replace('</td>','')
    line3 = line2.replace('</a>','')
    line3 = line3.replace('\n','')
    sub = pattern.search(line3)
    if sub:
        wf.write(line4)
        species_count = species_count + 1
        count = 1
        line4 = line3.replace(sub.group(),'')

    else:
        count = count + 1
        if (count != 2) | len(line3) > 2:
            line4 = line4 + ',' + line3
        if (count == 6):
            line4.strip()
            line4 = line4.replace('\'','')
            wf.write(line4+ '\n')
wf.write(line4)











rf.close()
wf.close()
