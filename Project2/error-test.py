def error_test(imgList, imgCluster):
    dict = {}
    for i in range(len(imgCluster)):
        key = imgCluster[i]
        dict[key] = []

    for i in range(len(imgCluster)):
        dict[imgCluster[i]].append(imgList[i])

    list3 = []
    error = 0
    for key in dict:
        list4 = dict[key]
        dict1 = {}
        for item in list4:
            val = item.split('_')[0]
            if val not in dict1.keys():
                dict1[val] = 1
            else:
                dict1[val] += 1
        maxcount = 0
        maxkey = ''
        for key in dict1:
            if dict1[key] > maxcount:
                maxcount = dict1[key]
                maxkey = key
        error += (len(list4) - maxcount)

    print("test numbers in total: ", len(imgList))
    print("error numbers in total: ", error)
    print("correct numbers in total: ", len(imgList) - error)
    print("error rate: ", (error / len(imgList)))
    print("correct rate: ", (1 - error / len(imgList)))
