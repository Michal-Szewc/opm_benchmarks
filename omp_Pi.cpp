#include <omp.h>
#include <stdlib.h>
#include <random>

#define SEED 1234;

#define L_ITERACJI 1000000000
#define THREADS 16

int main() {
	int i, w_kole = 0, result, threads;
	double x, y, start;
	std::random_device rd;  // a seed source for the random number engine
	std::mt19937 gen;		// random number generator
	std::uniform_real_distribution<double> distribution; // random number distribution

#if THREADS == 0
	#pragma omp parallel private(i, result, gen, distribution, x, y, threads)
#else
	#pragma omp parallel private(i, result, gen, distribution, x, y, threads) num_threads(THREADS)
#endif
	{
#pragma omp critical
		{
			std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
		}
		std::uniform_real_distribution<double> distribution(0.0, 1.0);

#if THREADS == 0
		threads = omp_get_num_threads();
#else
		threads = THREADS;
#endif

		// wait for all threads and one measures time
#pragma omp barrier
#pragma omp single
		{
			start = omp_get_wtime();
		}
	
		int result = 0;

		for (i = omp_get_thread_num(); i < L_ITERACJI; i += threads) {
			x = distribution(gen);
			y = distribution(gen);
			result += (x * x + y * y <= 1.0);
		}
#pragma omp critical
		w_kole += result;
	}
	double end = omp_get_wtime();

	double pi = ((double)(w_kole * 4.0) / (double)L_ITERACJI);
	printf("pi: %lf\n", pi);

	double time_s = (end - start);
	printf("elapsed time: %lf s\n", time_s);
}