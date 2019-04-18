#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "kmeans.h"

static inline float dist_2(Point a, Point b)
{
    return pow((a.x - b.x), 2) + pow((a.y - b.y), 2);
}

void KMeans_helper(Point *data, Point *means, int num_of_clusters, int num_of_iters, int num_of_elements, Point *sums, int *counts)
{
    for (int i = 0; i < num_of_iters; i++)
    {
        memset(sums, 0, num_of_clusters * sizeof(Point));
        memset(counts, 0, num_of_clusters * sizeof(int));

        for (int j = 0; j < num_of_elements; j++)
        {
            float min_dist = FLT_MAX;
            int my_cluster = 0;
            for (int c = 0; c < num_of_clusters; c++)
            {
                float dist = dist_2(data[j], means[c]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    my_cluster = c;
                }
            }
            sums[my_cluster].x += data[j].x;
            sums[my_cluster].y += data[j].y;
            counts[my_cluster] += 1;
        }

        for (int c = 0; c < num_of_clusters; c++)
        {
            if (counts[c] == 0)
                counts[c] = 1;
            means[c].x = sums[c].x / counts[c];
            means[c].y = sums[c].y / counts[c];
        }
    }
}

void KMeans(Point *data, Point *means, int num_of_clusters, int num_of_iters, int num_of_elements)
{
    Point *sums = (Point *)calloc(num_of_clusters, sizeof(Point));
    int *counts = (int *)calloc(num_of_clusters, sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < num_of_clusters; i++)
        means[i] = (data[rand() % num_of_elements]);

    KMeans_helper(data, means, num_of_clusters, num_of_iters, num_of_elements, sums, counts);
    free(sums);
    free(counts);
}