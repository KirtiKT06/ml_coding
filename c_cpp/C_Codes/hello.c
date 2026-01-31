#include<stdio.h>
/*int main()
{
    int i, n=5;
    for (i=0; i<=n; i++)
    {
        printf("%d\t", i*i);
    }
    return (0);
}*/
int main()
{
    int i, j;
    for(i=1; i<=5; i++)
    {
        for(j=1; j<=i; j++)
        {
            printf("%d", j);
        }
        printf("\n");
    }
}