import sys
sys.path.extend(["../", "./"])
from model import m_dict


print ("Current Model list output format: <key> ==> <Model>")


for key, value in m_dict.items():
    print (key+"===>",value)
