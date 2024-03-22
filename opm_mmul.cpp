#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

typedef long long int ll;

// range of pseudo random values used to generate random matrix [-VALUE_RANGE; VALUE_RANGE] (inclusive)
#define VALUE_RANGE 1000
// size of matricies
#define SIZE 1000
// number of threads ( set to 0 to use omp default )
#define THREADS 16

// print result
#define PRINT false
// use multithreading
#define USING_OPEN_MP true

// macros for easier value access in 1D array
#define A(i, j) a[ (i) * m_size + (j)]
#define B(i, j) b[ (i) * m_size + (j)]
#define C(i, j) c[ (i) * m_size + (j)]

// O(n) sqrt hehe
int sqrt(int n) {
	int res = 1;
	while (res * res <= n) ++res;
	return --res;
}

// simple matrix multiplication for even non-square matricies
void multiplyMatrix(ll const* a, int la, ll const* b, int lb, ll* c, int lc, int m_size) {
	for (int i = 0; i < la; ++i) {
		for (int j = 0; j < lb; ++j) {
			ll wyn = 0;
			for (int k = 0; k < lc; ++k) {
				wyn += A(i, k) * B(k, j);
			}
			// not a critical section, because threads save to different addresses
			C(i, j) += wyn;
		}
	}
}

// parallel matrix multiplication, onlu for square matricies
void parallelMultiplyMatrix(const ll* a, const ll* b, ll* c, int m_size) {
	// allocate memory for result
	ll* temp = (ll*)calloc(m_size * m_size, sizeof(ll));

	int block_num, block_size;
#if USING_OPEN_MP
	#if THREADS == 0
		#pragma omp parallel
	#else
		#pragma omp parallel num_threads(THREADS)
#endif
	{
		// number of blocks per matrix
		block_num = sqrt(omp_get_num_threads());
#else
	{
		// number of blocks based on potential number of threads
		block_num = sqrt(THREADS);
#endif
		block_num = block_num > m_size ? m_size : block_num;
		// length of block's smaller dimension
		block_size = m_size / block_num;
		// loop to offset memory access
		int i, j;
		for (int offest = 0; offest < block_num; ++offest) {
#if USING_OPEN_MP
	#pragma omp for private(i, j) firstprivate(m_size, block_size, a, b, temp) schedule(static, 1) nowait
#endif
			for (i = 0; i < block_num * block_num; ++i) {
				j = (i / block_num + i + offest) % block_num;
				multiplyMatrix(a + block_size * ((i / block_num) * m_size + j),
					(i / block_num == block_num - 1) ? m_size - block_size * (block_num - 1) : block_size,
					b + block_size * ((i % block_num) + j * m_size),
					(i % block_num == block_num - 1) ? m_size - block_size * (block_num - 1) : block_size,
					temp + block_size * ((i / block_num) * m_size + (i % block_num)),
					(j == block_num - 1) ? m_size - block_size * (block_num - 1) : block_size, m_size);
			}
		}
	}

	// copy from temp to c
	memcpy(c, temp, m_size * m_size * sizeof(ll));
	// free memory
	free(temp);
}

// return pointer to matrix of size [SIZE x SIZE] with random numbers from [-VALUE_RANGE; VALUE_RANGE]
ll* generate_matrix() {
	ll* res = (ll*)malloc(SIZE * SIZE * sizeof(ll));
	ll* end = res + SIZE * SIZE;

	for (ll* it = res; it != end; ++it) *it = rand() % (VALUE_RANGE * 2 + 1) - VALUE_RANGE;
	return res;
}

// prints matrix
void show(ll * c) {
	int m_size = SIZE;
	for (int i = 0; i < SIZE; ++i) {
		for (int j = 0; j < SIZE; ++j) {
			printf("%lld ", C(i, j));
		}
		printf("\n");
	}
}

int main(int argc, char* argv[])
{
	// get random seed
	srand((unsigned int)time(NULL));
	// generate matrix
	ll* m1 = generate_matrix(), * m2 = generate_matrix(), * c = (ll*)calloc(SIZE * SIZE, sizeof(ll));

	// calculate and measure time
	double start = omp_get_wtime();
	parallelMultiplyMatrix(m1, m2, c, SIZE);
	double end = omp_get_wtime();

	double time_s = (end - start);

#if PRINT
	printf("m1:\n");
	show(m1);
	printf("m2:\n");
	show(m2);
	printf("wyn:\n");
	show(c);
#endif

	printf("elapsed time: %lf s\n", time_s);

	// free memory
	free(m1);
	free(m2);
	free(c);
}
