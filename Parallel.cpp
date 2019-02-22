#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

double mul_vector_transpose_vector(double *v1, double *v2, int n){
	int i;
	double result = 0;
	#pragma omp parallel for reduction(+:result)
		for(i=0;i<n;i++){
			result += v1[i]*v2[i];
		}
	return result;
}

double* matrix_mul_vector(double **A, double *x, int n){
	int i,j;
	double *result = (double *)malloc(n*sizeof(double));
	#pragma omp parallel for
		for(i=0;i<n;i++){
			result[i] = mul_vector_transpose_vector(A[i],x,n);
		}
	return result;
}

double* negate_vector(double *g, int n){
	int i;
	double *s = (double *)malloc(n*sizeof(double));
	#pragma omp parallel for
		for(i=0;i<n;i++){
			s[i] = (-1)*g[i];
		}
	return s;
}

void copy_vector(double *v1, double *v2, int n){
	int i;
	#pragma omp parallel for
		for(i=0;i<n;i++){
			v1[i] = v2[i];
		}
}

double* scalar_mul_vector(double *s, double c, int n){
	int i;
	double *result = (double *)malloc(n*sizeof(double));
	#pragma omp parallel for
		for(i=0;i<n;i++){
			result[i] = c*s[i];
		}
	return result;
}

double* vector_add_vector(double *v1, double *v2, int n){
	int i;
	double *result = (double *)malloc(n*sizeof(double));
	#pragma omp parallel for
		for(i=0;i<n;i++){
			result[i] = v1[i] + v2[i];
		}
	return result;
}

double* vector_sub_vector(double *v1, double *v2, int n){
	int i;
	double *result = (double *)malloc(n*sizeof(double));
	#pragma omp parallel for
		for(i=0;i<n;i++){
			result[i] = v1[i] - v2[i];
		}
	return result;
}

int main(){
	int n,i,j,t = 1000;
	printf("Enter the size of matrix \n");
	scanf("%d",&n);
	double **A = (double **)malloc(n*sizeof(double *));
	double *b = (double *)malloc(n*sizeof(double));
	double *x = (double *)malloc(n*sizeof(double));
	for(i=0;i<n;i++){
		A[i] = (double *)malloc(n*sizeof(double));
	}
	printf("Enter the elements of Matrix A row wise \n");
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			scanf("%lf",&A[i][j]);
		}
	}
	printf("Enter the elements of Vector b \n");
	for(i=0;i<n;i++){
		scanf("%lf",&b[i]);
	}
	printf("Enter the start Vector x \n");
	for(i=0;i<n;i++){
		scanf("%lf",&x[i]);
	}
	clock_t start,end;
	start = clock();
	double *Ax = matrix_mul_vector(A,x,n);
	double *g = vector_sub_vector(Ax,b,n);
	double *s = negate_vector(g,n);
	double numerator = (-1)*mul_vector_transpose_vector(s,g,n);
	double *tmp = matrix_mul_vector(A,s,n);
	double denominator = mul_vector_transpose_vector(s,tmp,n);
	double gamma = (numerator)/(denominator);
	double *s1 = scalar_mul_vector(s,gamma,n);
	x = vector_add_vector(x,s1,n);
	while(t--){
		Ax = matrix_mul_vector(A,x,n);
		double *gnew = vector_sub_vector(Ax,b,n);
		numerator = mul_vector_transpose_vector(gnew,gnew,n);
		denominator = mul_vector_transpose_vector(g,g,n);
		double beta = (numerator)/(denominator);
		s1 = scalar_mul_vector(s,beta,n);
		double *neggnew = negate_vector(gnew,n);
		s = vector_add_vector(s1,neggnew,n);
		numerator = mul_vector_transpose_vector(s,neggnew,n);
		tmp = matrix_mul_vector(A,s,n);
		denominator = mul_vector_transpose_vector(s,tmp,n);
		gamma = (numerator)/(denominator);
		s1 = scalar_mul_vector(s,gamma,n);
		x = vector_add_vector(x,s1,n);
		copy_vector(g,gnew,n);
	}
	end = clock();
	printf("Solution is x = [");
	for(i=0;i<n-1;i++){
		printf(" %lf,",x[i]);
	}
	printf(" %lf ]\n",x[n-1]);
	printf("Time taken for parallel execution is %f \n",(double)(end - start)/CLOCKS_PER_SEC); 
	return 0;
}
