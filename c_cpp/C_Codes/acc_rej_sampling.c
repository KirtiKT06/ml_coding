// code to implement inversion sampling for f(x)=2x, x \in [0,1]
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
int main()
{
    int accepted=0, trials=0, n=1000000;
    double x, y;
    srand(time(NULL));
     FILE *fptr;
    fptr = fopen("../data/acc_rej_2x.dat", "w");
    if (fptr == NULL) 
    {
        printf("Error opening file.\n");
        return 1;
    } 
    while (accepted<n)
    {
        x = (double) rand()/RAND_MAX;
        y = 2.0*(double) rand()/RAND_MAX;
        trials++;
        if (y<=2*x)
        {
            fprintf(fptr, "%lf\n", x);
            accepted++;
        }
    }
    fclose(fptr);
    printf("Acceptance ratio = %lf\n", (double)accepted/trials);
    return 0;
}