#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ctime>
#include <string>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
//#include "scoped_array.h"
#else
#include "CL/opencl.h"
#include "AOCL_Utils.h"
using namespace aocl_utils;
#endif
#ifdef __APPLE__
// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 1;
cl_device_id device; // num_devices elements
cl_context context = NULL;
cl_command_queue queue; // num_devices elements
cl_program program = NULL;
cl_kernel kernel0; // num_devices elements
cl_kernel kernel1;
#else
// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel0; // num_devices elements
scoped_array<cl_kernel> kernel1; // num_devices elements
#endif

#define MAX_GROUP_SIZE 256

#ifdef __APPLE__
static int LoadTextFromFile(const char *file_name, char **result_string, size_t *string_len);
#define LOCAL_MEM_SIZE 1024
void _checkError(int line,
                 const char *file,
                 cl_int error,
                 const char *msg,
                 ...);

#define checkError(status, ...) _checkError(__LINE__, __FILE__, status, __VA_ARGS__)
#endif

bool init_opencl();

int fpga_kmeans(Point *data, Point *means, int num_of_clusters, int num_of_iters, int num_of_elements)
{
    init_opencl();

    int err;
    cl_mem d_data, d_means, d_sums, d_counts;
    cl_int status;

    int num_of_groups = num_of_elements / MAX_GROUP_SIZE;

    d_data = clCreateBuffer(context, CL_MEM_READ_ONLY, num_of_elements * sizeof(Point), NULL, &status);
    d_means = clCreateBuffer(context, CL_MEM_READ_WRITE, num_of_clusters * sizeof(Point), NULL, &status);
    d_sums = clCreateBuffer(context, CL_MEM_READ_WRITE, num_of_clusters * num_of_groups * sizeof(Point), NULL, &status);
    d_counts = clCreateBuffer(context, CL_MEM_READ_WRITE, num_of_clusters * num_of_groups * sizeof(int), NULL, &status);

    srand(time(NULL));
    for (int i = 0; i < num_of_clusters; i++)
        means[i] = (data[rand() % num_of_elements]);
    clEnqueueWriteBuffer(queue, d_data, CL_FALSE, 0, num_of_elements * sizeof(Point), data, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_means, CL_FALSE, 0, num_of_clusters * sizeof(Point), means, 0, NULL, NULL);
    clFinish(queue);

    cl_event kernel_event;
    unsigned argi;
    for (int i = 0; i < num_of_iters; i++)
    {
        //assign
        argi = 0;
        clSetKernelArg(kernel0, argi++, sizeof(cl_mem), &d_data);
        clSetKernelArg(kernel0, argi++, sizeof(cl_mem), &d_means);
        clSetKernelArg(kernel0, argi++, sizeof(cl_mem), &num_of_clusters);
        clSetKernelArg(kernel0, argi++, sizeof(cl_mem), &d_sums);
        clSetKernelArg(kernel0, argi++, sizeof(cl_mem), &d_counts);
        clSetKernelArg(kernel0, argi++, MAX_GROUP_SIZE * sizeof(int), NULL);
        clSetKernelArg(kernel0, argi++, 2 * MAX_GROUP_SIZE * sizeof(float), NULL);
        size_t global_work_size[1] = {num_of_elements};
        size_t local_work_size[1] = {MAX_GROUP_SIZE};
        clEnqueueNDRangeKernel(queue, kernel0, 1, NULL, global_work_size, local_work_size, 0, NULL, &kernel_event);
        clFinish(queue);

        // update means
        argi = 0;
        clSetKernelArg(kernel1, argi++, sizeof(cl_mem), &d_means);
        clSetKernelArg(kernel1, argi++, sizeof(cl_mem), &d_sums);
        clSetKernelArg(kernel1, argi++, sizeof(cl_mem), &d_counts);
        clSetKernelArg(kernel1, argi++, sizeof(int), &num_of_clusters);
        clSetKernelArg(kernel1, argi++, sizeof(int), &num_of_groups);
        global_work_size[0] = num_of_clusters;
        clEnqueueNDRangeKernel(queue, kernel1, 1, NULL, global_work_size, NULL, 0, NULL, &kernel_event);
        clFinish(queue);
    }

    clReleaseEvent(kernel_event);
    clEnqueueReadBuffer(queue, d_means, CL_TRUE, 0, num_of_clusters * sizeof(Point), means, 0, NULL, NULL);
    clFinish(queue);

    if (d_data)
        clReleaseMemObject(d_data);
    if (d_means)
        clReleaseMemObject(d_means);
    if (d_sums)
        clReleaseMemObject(d_sums);
    if (d_counts)
        clReleaseMemObject(d_counts);
    return 0;
}

// Initializes the OpenCL objects.
bool init_opencl()
{
    int err;
    cl_int status;

    // printf("Initializing OpenCL\n");
#ifdef __APPLE__
    int gpu = 1;
    cl_device_id devices[2];
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 2, devices, NULL);
    device = devices[1]; // 0 - Intel, 1 - AMD

    // size_t max_work_gp;
    // clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_gp), &max_work_gp, NULL);
    // printf("device max group size: %zu\n", max_work_gp);

    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    // Create the context.
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkError(status, "Failed to create context");
#else
    if (!setCwdToExeDir())
    {
        return false;
    }

    // Get the OpenCL platform.
    platform = findPlatform("Altera");
    if (platform == NULL)
    {
        printf("ERROR: Unable to find Altera OpenCL platform.\n");
        return false;
    }

    // Query the available OpenCL device.
    device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    printf("Platform: %s\n", getPlatformName(platform).c_str());
    printf("Using %d device(s)\n", num_devices);
    for (unsigned i = 0; i < num_devices; ++i)
    {
        printf("  %s\n", getDeviceName(device[i]).c_str());
    }
    // Create the context.
    context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
    checkError(status, "Failed to create context");

    // size_t max_work_gp;
    // clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_gp), &max_work_gp, NULL);
    // printf("device max group size: %zu\n", max_work_gp);
#endif

    // Create the program for all device. Use the first device as the
    // representative device (assuming all device are of the same type).
#ifndef __APPLE__
    std::string binary_file = getBoardBinaryFile("kmeans", device[0]);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    //Create per-device objects.
    queue.reset(num_devices);
    kernel0.reset(num_devices);
    kernel1.reset(num_devices);
    for (unsigned i = 0; i < num_devices; ++i)
    {
        // Command queue.
        queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
        checkError(status, "Failed to create command queue");

        // Kernel.
        const char *kernel_name = "find_nearest";
        kernel0[i] = clCreateKernel(program, kernel_name, &status);
        checkError(status, "Failed to create kernel0");

        kernel_name = "update_means";
        kernel1[i] = clCreateKernel(program, kernel_name, &status);
        checkError(status, "Failed to create kernel1");
    }
#else
    char *source = 0;
    size_t length = 0;
    LoadTextFromFile("kmeans.cl", &source, &length);
    const char *kernel_name = "find_nearest";
    program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        char *buff_erro;
        cl_int errcode;
        size_t build_log_len;
        errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
        if (errcode)
        {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-1);
        }

        buff_erro = (char *)malloc(build_log_len);
        if (!buff_erro)
        {
            printf("malloc failed at line %d\n", __LINE__);
            exit(-2);
        }

        errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
        if (errcode)
        {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-3);
        }

        fprintf(stderr, "Build log: \n%s\n", buff_erro); //Be careful with  the fprint
        free(buff_erro);
        fprintf(stderr, "clBuildProgram failed\n");
        exit(EXIT_FAILURE);
    }
    checkError(status, "Failed to build program");

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    kernel0 = clCreateKernel(program, kernel_name, &status);
    kernel_name = "update_means";
    kernel1 = clCreateKernel(program, kernel_name, &status);

#endif
    return true;
}

void cleanup()
{
#ifndef __APPLE__
    for (unsigned i = 0; i < num_devices; ++i)
    {
        if (kernel0 && kernel0[i])
        {
            clReleaseKernel(kernel0[i]);
        }
        if (kernel1 && kernel1[i])
        {
            clReleaseKernel(kernel1[i]);
        }
        if (queue && queue[i])
        {
            clReleaseCommandQueue(queue[i]);
        }
    }
#else
    clReleaseKernel(kernel0);
    clReleaseKernel(kernel1);
    clReleaseCommandQueue(queue);
#endif
    if (program)
        clReleaseProgram(program);
    if (context)
        clReleaseContext(context);
}
#ifdef __APPLE__
static int LoadTextFromFile(
    const char *file_name, char **result_string, size_t *string_len)
{
    int fd;
    unsigned file_len;
    struct stat file_status;
    int ret;

    *string_len = 0;
    fd = open(file_name, O_RDONLY);
    if (fd == -1)
    {
        printf("Error opening file %s\n", file_name);
        return -1;
    }
    ret = fstat(fd, &file_status);
    if (ret)
    {
        printf("Error reading status for file %s\n", file_name);
        return -1;
    }
    file_len = file_status.st_size;

    *result_string = (char *)calloc(file_len + 1, sizeof(char));
    ret = read(fd, *result_string, file_len);
    if (!ret)
    {
        printf("Error reading from file %s\n", file_name);
        return -1;
    }

    close(fd);

    *string_len = file_len;
    return 0;
}

// High-resolution timer.
double getCurrentTimestamp()
{
#ifdef _WIN32 // Windows
    // Use the high-resolution performance counter.

    static LARGE_INTEGER ticks_per_second = {};
    if (ticks_per_second.QuadPart == 0)
    {
        // First call - get the frequency.
        QueryPerformanceFrequency(&ticks_per_second);
    }

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);

    double seconds = double(counter.QuadPart) / double(ticks_per_second.QuadPart);
    return seconds;
#else // Linux
    timespec a;
    clock_gettime(CLOCK_MONOTONIC, &a);
    return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
#endif
}

void _checkError(int line,
                 const char *file,
                 cl_int error,
                 const char *msg,
                 ...)
{
    // If not successful
    if (error != CL_SUCCESS)
    {
        // Print line and file
        printf("ERROR: ");
        printf("\nLocation: %s:%d\n", file, line);

        // Print custom message.
        va_list vl;
        va_start(vl, msg);
        vprintf(msg, vl);
        printf("\n");
        va_end(vl);

        // Cleanup and bail.
        cleanup();
        exit(error);
    }
}
#endif