#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <string>

#define MKL
#ifdef MKL
#include "mkl.h"
#include "omp.h"
#endif

using namespace std;

void generation(double* mat, size_t size)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> uniform_distance(-2.001, 2.001);
	for (size_t i = 0; i < size * size; i++)
		mat[i] = uniform_distance(gen);
}
/*
void matrix_mult(double* a, double* b, double* res, size_t size)
{
	int size_block = 32;

#pragma omp parallel for
	for (int ib = 0; ib < size; ib += size_block)
	{
		for (int jb = 0; jb < size; jb += size_block)
		{
			for (int kb = 0; kb < size; kb += size_block)
			{
				for (int i = ib; i < size && i < ib + size_block; ++i) 
				{
					for (int j = jb; j < size && j < jb + size_block; ++j)
					{
						double tmp = a[i * size + j];
						int local_min_k = min(kb + size_block, static_cast<int>(size));
						for(int k = kb; k < local_min_k; k+=4)
						{
							__m256d vector_b = _mm256_loadu_pd(&b[j * size + k]);

							__m256d vector_res = _mm256_loadu_pd(&res[i * size + k]);

							vector_res = _mm256_fmadd_pd(_mm256_set1_pd(tmp), vector_b, vector_res);

							_mm256_storeu_pd(&res[i * size + k], vector_res);
						}
					}
				}
			}
		}
	}
}*/

void matrix_mult(double* a, double* b, double* res, size_t size)
{
	int size_block = 20;

#pragma omp parallel for
	for (int ib = 0; ib < size; ib += size_block)
	{
		for (int jb = 0; jb < size; jb += size_block)
		{
			for (int kb = 0; kb < size; kb += size_block)
			{
				for (int i = ib; i < ib + size_block; ++i)
				{
					for (int j = jb; j < jb + size_block; ++j)
					{
						__m256d vector_a = _mm256_set1_pd(a[i * size + j]);
						for (int k = kb; k < kb + size_block; k += 4)
						{
							__m256d vector_b = _mm256_loadu_pd(&b[j * size + k]);
							__m256d vector_res = _mm256_loadu_pd(&res[i * size + k]);

							vector_res = _mm256_fmadd_pd(vector_a, vector_b, vector_res);

							_mm256_storeu_pd(&res[i * size + k], vector_res);
						}
					}
				}
			}
		}
	}
}

int main()
{
	double* mat, * mat_mkl, * a, * b, * a_mkl, * b_mkl;
	size_t size = 1000;
	chrono::time_point<chrono::system_clock> start, end;

	mat = new double[size * size];
	a = new double[size * size];
	b = new double[size * size];
	generation(a, size);
	generation(b, size);
	memset(mat, 0, size * size * sizeof(double));

#ifdef MKL     
	mat_mkl = new double[size * size];
	a_mkl = new double[size * size];
	b_mkl = new double[size * size];
	memcpy(a_mkl, a, sizeof(double) * size * size);
	memcpy(b_mkl, b, sizeof(double) * size * size);
	memset(mat_mkl, 0, size * size * sizeof(double));
#endif

	start = chrono::system_clock::now();
	matrix_mult(a, b, mat, size);
	end = chrono::system_clock::now();


	int elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time: " << elapsed_seconds / 1000.0 << " sec" << endl;

#ifdef MKL 
	start = chrono::system_clock::now();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		size, size, size, 1.0, a_mkl, size, b_mkl, size, 0.0, mat_mkl, size);
	end = chrono::system_clock::now();

	elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time mkl: " << elapsed_seconds / 1000.0 << " sec" << endl;

	int flag = 0;
	for (unsigned int i = 0; i < size * size; i++)
		if (abs(mat[i] - mat_mkl[i]) > size * 1e-14) {
			flag = 1;
		}
	if (flag)
		cout << "fail" << endl;
	else
		cout << "correct" << endl;

	delete (a_mkl);
	delete (b_mkl);
	delete (mat_mkl);
#endif

	delete (a);
	delete (b);
	delete (mat);

	//system("pause");

	return 0;
}
