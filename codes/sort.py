# https://www.geeksforgeeks.org/python-indices-of-n-largest-elements-in-list/

# https://kanoki.org/2020/01/14/find-k-smallest-and-largest-values-and-its-indices-in-a-numpy-array/

# Python3 code to demonstrate working of 
# Indices of N largest elements in list 
# using sorted() + lambda + list slicing 
  
# initialize list 
test_list = [5, 6, 10, 4, 7, 1, 19] 
  
# printing original list 
print("The original list is : " + str(test_list)) 
  
# initialize N  
N = 4
  
# Indices of N largest elements in list 
# using sorted() + lambda + list slicing 
res = sorted(range(len(test_list)), key = lambda sub: test_list[sub])[-N:] 
  
# printing result 
print("Indices list of max N elements is : " + str(res)) 



import numpy as np
# arr=np.random.randint(0,100,size=10)
# print(arr)

def find_index(list, element):
    indices = [i for i, v in enumerate(list) if v == element]

    return indices

def find_n_index(arr, n, largest=False):
    if largest: 
        res = np.partition(arr,-4)[-4:]
        res.sort()
        res = res[::-1]
    else:
        # get the first 4 smallest values
        res = np.partition(arr,n)[:n]
        res.sort()

    # get index
    indexs = []
    for r in res:
        ans = find_index(arr, r)
        for i in ans:
            indexs.append(i)
        if len(ans) != 1:
            arr = arr[len(ans):]
    return indexs

# largest is sorted, however smallest doesn't

#arr = [10,20,50,30,40,70,60,90,80]
#res = arr[np.argpartition(arr,5)[:5]]
#print(res)

#               0. 1. 2. 3. 4. 5. 6. 7. 8
arr= np.array([60,90,10,10,50,30,40,70,80])
print(np.argpartition(arr,5)[:5])
res = arr[np.argpartition(arr,5)[:5]]
res.sort()
print(res)

# expected result: [2, 3, 5, 6, 4]

# [0,1,2,3]
#print(find_n_index(arr,4, largest=True))

# given input
#               0. 1. 2. 3. 4. 5. 6. 7. 8
arr= np.array([60,90,10,20,50,30,40,70,80])

# indices of 5 smallest un-sorted elements
small_indices= np.argpartition(arr,5)[:5]
print("DEBUG small_indices", small_indices)
print("DEBUG arr[small_indices]", arr[small_indices])

d = {}
for i in small_indices:
    d[i] = arr[i]
print("DEBUG d", d)

# sort dict by value instead of key
sd = sorted(d.items(), key=lambda item: item[1])
print("DEBUG sd", sd)

ans = []
for ele in sd:
    ans.append(ele[0])
print("DEBUG ans", ans)

print("expected result: [2, 3, 5, 6, 4]")

# x = [('sampl', 'NN'), ('eleg', 'NN'), ('typewallpap', 'NN'), ('technolog', 'NN'), ('hangexpand', 'NN'), ('clothproduc', 'NN'), ('damp', 'JJ'), ('embellish', 'JJ'), ('sadi', 'NN'), ('vinylwash', 'NN'), ('gold', 'NN'), ('past', 'IN'), ('return', 'NN'), ('kingdomstraight', 'VBD'), ('floral', 'JJ'), ('make', 'NN'), ('pattern', 'NN'), ('easi', 'FW'), ('curvac', 'NN'), ('unit', 'NN'), ('ar', 'JJ'), ('gentl', 'NN'), ('featur', 'NN'), ('silver', 'NN'), ('trail', 'VBP'), ('wall', 'NN'), ('underst', 'JJ')]

# for i in x:
#     print(i[1])


