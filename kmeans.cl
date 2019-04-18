typedef struct Point
{
    float x, y;
} Point;

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void find_nearest(__global Point* restrict data, __global Point* restrict means, const int num_of_clusters, __global Point* restrict sums, __global int* restrict counts, __local int* restrict local_counts, __local float* restrict local_data)
{
    int index = get_global_id(0);
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    float min_dist = FLT_MAX;
    int my_cluster = 0;
    float x = data[index].x;
    float y = data[index].y;

    for (int c = 0; c < num_of_clusters; c++)
    {
        float dist = (x - means[c].x) * (x - means[c].x) + (y - means[c].y) * (y - means[c].y);
        if (dist < min_dist)
        {
            min_dist = dist;
            my_cluster = c;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int loc_y = local_id + group_size;

    for (int c = 0; c < num_of_clusters; c++)
    {
        if (my_cluster == c)
        {
            local_data[local_id] = x;
            local_data[loc_y] = y;
            local_counts[local_id] = 1;
        }
        else
        {
            local_data[local_id] = 0.0f;
            local_data[loc_y] = 0.0f;
            local_counts[local_id] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = group_size / 2; stride > 0; stride /= 2)
        {
            if (local_id < stride)
            {
                local_data[local_id] += local_data[local_id + stride];
                local_data[loc_y] += local_data[loc_y + stride];
                local_counts[local_id] += local_counts[local_id + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (local_id == 0)
        {
            int loc = c + group_id * num_of_clusters;
            sums[loc].x = local_data[0];
            sums[loc].y = local_data[group_size];
            counts[loc] = local_counts[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void update_means(__global Point* restrict means,
                           __global Point* restrict sums,
                           __global int* restrict counts,
                           const int num_of_clusters,
                           const int num_of_groups)
{
    int index = get_global_id(0);
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    int count = 0;

    for (int i = 0; i < num_of_groups; i++)
    {
        int loc = index + i * num_of_clusters;
        sum_x += sums[loc].x;
        sum_y += sums[loc].y;
        count += counts[loc];
    }

    if (count == 0)
        count = 1;
    means[index].x = sum_x / count;
    means[index].y = sum_y / count;
}