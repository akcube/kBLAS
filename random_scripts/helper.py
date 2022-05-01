import math
from hurry.filesize import size

EL_SIZE = 8

def vector(s:int) -> (int, int) :
	return (1, s//EL_SIZE)

# y = Ax + y
# m*n + m + n
def matrix_vector(s:int) -> (int, int) :
	s = s//EL_SIZE
	n = math.isqrt(s)
	mindist = abs(s - (n * n + 2 * n))
	best = (n, n)
	for i in range(n-20, n+20):
		for j in range(n-20, n+20):
			val = abs(s - ((i * j) + i + j));
			if(val < mindist):
				mindist = val
				best = (i, j)
	return best

def matrix_matrix(s:int) -> (int, int) :
	s = s//EL_SIZE
	s = s//2
	return (math.isqrt(s), math.isqrt(s))

sizes_kb = [50, 100, 300, 600, 1200]
sizes_mb = [2, 5, 8, 12, 16, 20, 32, 64, 128, 256, 512]
sizes_gb = [1]

sizes = []
for s in sizes_kb:
	sizes.append(s * 1024)
for s in sizes_mb:
	sizes.append(s * 1024 * 1024)
for s in sizes_gb:
	sizes.append(s * 1024 * 1024 * 1024)

type = int(input("1 - vector\n2 - matrix vector\n3 - matrix matrix\n"))

res = ""

for s in sizes:
	if type == 1:
		pp = size(s)
		print(pp, end='')
		print(": ", vector(s))
	elif type == 2:
		pp = size(s)
		print(pp, end='')
		r = matrix_vector(s)
		print(": ", r)

		res += "{{1, 1}, {%d, %d}, {%d, 1}, {1, 1}, {%d, 1}}, " % (r[0], r[1], r[1], r[1])

	elif type == 3:
		pp = size(s)
		print(pp, end='')
		print(": ", matrix_matrix(s))

print(res)