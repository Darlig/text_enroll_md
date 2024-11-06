import matplotlib.pyplot as plt
import numpy as np
import sys
import json

in_score = sys.argv[1]
out_png = sys.argv[2]
keywords = [0, 1, 2, 3, 4, 5, 6]

for keyword in keywords:
    scores = []
    with open(in_score) as f:
        for line in f.readlines():
            line_dict = json.loads(line.strip())
            if line_dict['target_keyword'] == keyword:
                utt_score = [ chunk_score[keyword] for chunk_score in line_dict['scores'] ]
                utt_score = max(utt_score)
                # print(len(line_dict['scores'][0]))
                scores.append(utt_score)

    # # 生成示例数据
    # data = np.random.randn(1000)
    # for i in range(10):
    #     print(data[i])

    # 使用 matplotlib 绘制直方图
    plt.figure()
    plt.hist(scores, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig("{}_{}.png".format(out_png, keyword))



