
import math
import numpy as np
import sys
from sklearn.decomposition import PCA
#start = timeit.default_timer()

doc_count  = 0

total_termD = {}

file_name = sys.argv[1]

f_count_doc = open(file_name)


with open (file_name,"r") as f:
	for line in f:
		doc_count = doc_count + 1
	'''
	count = list(range(0,doc_count-2))
	'''
count = doc_count 


f = open(file_name,'r')
data = f.read()
words = data.split()
f.close()
total_termD = {}

for word in words:
	if word not in total_termD:
		total_termD[word] = 0
	total_termD[word] += 1

count_num_doc = total_termD.copy();

for key,value in count_num_doc.items():
	count_num_doc[key] = 0

with open (file_name,"r") as f:
	for line in f:
		for key,value in count_num_doc.items():
			if key in line.split():
				count_num_doc[key] = count_num_doc[key] + 1
		
idf = count_num_doc.copy();

for key,value in idf.items():
	idf[key] = math.log(doc_count/count_num_doc[key])


tfidf_matrix = np.zeros((doc_count,len(total_termD)))
pos_i = 0

with open (file_name,"r") as f:
	for line in f:
		
		total_termD_app = total_termD.copy()

		
		for key,value in total_termD_app.items():
			total_termD_app[key] = 0
		
		
		for key,value in total_termD_app.items():
			total_termD_app[key] = line.split().count(key)

		tf = {}
		for key,value in total_termD_app.items():
			tf[key] = total_termD_app[key]/len(line.split())

		tfidf = {}
		for key,value in tf.items():
			tfidf[key] = tf[key]*idf[key]
		pos_j = 0
		for key, value in tfidf.items():
			tfidf_matrix[pos_i,pos_j] = round(tfidf[key],3)
			pos_j = pos_j + 1
		pos_i = pos_i + 1
'''
		count_word = 0
		if count_1 >= 2:
			for word in line.split():
				
				if word:
					count_word = count_word + 1
					if word in total_termD_app:
						total_termD_app[word] += line.count(word)
					else:
						total_termD_app[word] = line.count(word)
		count_1 = count_1 - 1
		tf = {}
		for key,value in total_termD_app.items():
			tf[key] = total_termD_app[key]/count_word;
'''


print (len(total_termD))
man_dist = np.zeros((doc_count,1))
for i in range(doc_count):
	for j in range(len(total_termD)):
		man_dist[i,0] = man_dist[i,0] + abs(tfidf_matrix[i,j] - tfidf_matrix[499,j])
 
#cp_man_dist = np.copy(man_dist)


a = man_dist[0,0]
for i in range(5):
	counter = 0
	index = 0
	for i in man_dist:
		if i < a:
			a = i
			index = counter
		counter = counter + 1 
	print (index+1, end = ' ')
	man_dist[index,0] = 10000

print ('')
'''
	if i != 5:
		print (np.argmin(cp_man_dist),end = ' ')
		cp_man_dist[np.argmin(cp_man_dist),0] = 10000
	else:
		print (np.argmin(cp_man_dist))
'''
euc_dist = np.zeros((doc_count,1))

for i in range(doc_count):
	for j in range(len(total_termD)):
		euc_dist[i,0] = euc_dist[i,0] + (tfidf_matrix[i,j] - tfidf_matrix[499,j])*(tfidf_matrix[i,j] - tfidf_matrix[499,j])

for i in range(doc_count):
	
		euc_dist[i,0] = (euc_dist[i,0])**0.5

#cp_euc_dist = np.copy(euc_dist)

a = euc_dist[0,0]
for i in range(5):
	counter = 0
	index = 0
	for i in euc_dist:
		if i < a:
			a = i
			index = counter
		counter = counter + 1 
	print (index+1, end = ' ')
	euc_dist[index,0] = 10000
print('')
'''
for i in range(5):
	if i != 5:
		print(np.argmin(cp_euc_dist),end = ' ')
		cp_euc_dist[np.argmin(cp_euc_dist),0] = 10000
	else:
		print (np.argmin(cp_euc_dist))
'''

sup_dist = np.zeros((doc_count,1))
for i in range(doc_count):
	temp = np.zeros((1,len(total_termD)))
	for j in range(len(total_termD)):
		temp[0,j] = abs(tfidf_matrix[i,j] - tfidf_matrix[499,j])
	t_max = temp[0,0]
	for x in range(len(total_termD)):
		if temp[0,x] > t_max:
			t_max = temp[0,x]
	sup_dist[i,0] = t_max
#cp_sup_dist = np.copy(sup_dist)
a = sup_dist[0,0]
for i in range(5):
	counter = 0
	index = 0
	for i in sup_dist:
		if i < a:
			a = i
			index = counter
		counter = counter + 1 
	print (index+1, end = ' ')
	sup_dist[index,0] = 10000
print('')

cos_dist = np.zeros((doc_count,1))

for i in range(doc_count):
	sump = 0
	suma = 0
	sumb = 0
	for j in range(len(total_termD)):
		sump = sump + (tfidf_matrix[i,j])*(tfidf_matrix[499,j])
		suma = suma + (tfidf_matrix[i,j])*(tfidf_matrix[i,j])
		sumb = sumb + (tfidf_matrix[499,j])*(tfidf_matrix[499,j])
	cos_dist[i,0] = sump/((suma**(0.5))*(sumb**(0.5)))

#cp_cos_dist = np.copy(cos_dist)

a_t = 0
for i in range(5):
	counter = 0
	index = 0
	for j in cos_dist:
		if j > a_t:
			a_t = j
			index = counter
		counter = counter + 1 
	print (index+1, end = ' ')
	cos_dist[index,0] = 0
print('')

tfidf_matrix_copy = tfidf_matrix.copy()
pca = PCA(n_components=2, svd_solver = 'full')

#print(pca.fit_transform(tfidf_matrix_copy))

pca_matrix = pca.fit_transform(tfidf_matrix_copy)


euc_dist_pca = np.zeros((doc_count,1))

for i in range(doc_count):
	for j in range(2):
		euc_dist_pca[i,0] = euc_dist_pca[i,0]+(pca_matrix[i,j] - pca_matrix[499,j])*(pca_matrix[i,j] - pca_matrix[499,j])

for i in range(doc_count):
	euc_dist_pca[i,0] = (euc_dist_pca[i,0])**(0.5)



#cp_euc_dist_pca = euc_dist_pca.copy()
a = euc_dist_pca[0,0]
for i in range(5):
	counter = 0
	index = 0
	for i in euc_dist_pca:
		if i < a:
			a = i
			index = counter
		counter = counter + 1 
	print (index+1, end = ' ')
	euc_dist_pca[index,0] = 10000

print('')


#print (pca)


#stop = timeit.default_timer()

#print (stop - start)

 
















