// code to implement inversion sampling for f(x)=2x, x \in [0,1]
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double invrs(double u)
    {
        return sqrt(u);
    }
int main()
{
    int i, n = 1000000;
    double x, y;
    srand(42);
    FILE *fptr;
    fptr = fopen("../data/inv_2x.dat", "w");
    if (fptr == NULL) 
    {
        printf("Error opening file.\n");
        return 1;
    } 
    for(i=0; i<n; i++)
    {
        x = (double) rand()/RAND_MAX;
        y = invrs(x);
        fprintf(fptr, "%lf\n", y);
    }
    fclose(fptr);
    return 0;
}
