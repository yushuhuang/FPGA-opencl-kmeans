#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "kmeans.h"

#define num_of_elements 1024
#define num_of_clusters 3
#define num_of_iters 300

int main(int argc, char **argv)
{
    struct timeval time_start, time_end;

    Point *data = (Point *)malloc(num_of_elements * sizeof(Point));
    FILE *fp = fopen("data", "r");
    int i = 0;
    while (fscanf(fp, "%f %f", &data[i].x, &data[i].y) == 2)
        i += 1;
    fclose(fp);

    Point *means = (Point *)malloc(num_of_clusters * sizeof(Point));

    int mode = 0;
    if (argc == 2)
        mode = atoi(argv[1]);
    switch (mode)
    {
    case 1:
        gettimeofday(&time_start, NULL);
        fpga_kmeans(data, means, num_of_clusters, num_of_iters, num_of_elements);
        gettimeofday(&time_end, NULL);
        fprintf(stderr, "FPGA: %ld\n", ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
        break;
    default:
        gettimeofday(&time_start, NULL);
        KMeans(data, means, num_of_clusters, num_of_iters, num_of_elements);
        gettimeofday(&time_end, NULL);
        fprintf(stderr, "CPU: %ld\n", ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
        break;
    }

    fp = fopen("means", "w");

    for (int i = 0; i < 3; i++)
        fprintf(fp, "%f %f\n", means[i].x, means[i].y);

    free(data);
    free(means);
    return 0;
}