from pyhanlp import *

#h = HanLP.convertToPinyinString("你", ' ', True)
# https://github.com/hankcs/HanLP/blob/1.x/src/main/java/com/hankcs/hanlp/HanLP.java#L55
# https://github.com/hankcs/HanLP.git
s = '安宁你想好了吗尼日利亚这个事情就这么不了了之么看着现场着火啦'
segment = HanLP.segment(s)
for term in segment:
    init = HanLP.convertToPinyinFirstCharString(term.word, ',', True)
    full = HanLP.convertToPinyinList(term.word)
    print (init, full)
#print (segment)
#h = HanLP.convertToPinyinList('你想好了吗？ 这个事情就这么不了了之么')
##print (HanLP.Config.CoreDictionaryPath)
#print (h)
#